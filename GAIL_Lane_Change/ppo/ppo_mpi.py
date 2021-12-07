from baselines.common import dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque
from common_util.train_util import traj_segment_generator, add_vtarg_and_adv, timed, allmean
from common_util.statistics import stats
from common_util.runner import Runner
import os


def learn(env, policy_fn, reward_giver, expert_dataset, rank, pi_nn,
        pretrained, pretrained_weight, *,
        d_step, max_grad_norm,
        clip_param, entcoeff, save_per_iter,# clipping parameter epsilon, entropy coeff
        ckpt_dir, log_dir, timesteps_per_actorbatch, task_name,
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, d_stepsize=3e-4,# advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5, load_model_path,
        schedule='constant' # annealing for stepsize parameters (epsilon and adam)
        ):

    nworkers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    np.set_printoptions(precision=3)
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    if pi_nn == "gru" or pi_nn == "lstm":
        pi = policy_fn(pi_nn, "pi", ob_space, ac_space, optim_batchsize, reuse=(pretrained_weight != None)) # Construct network for new policy
        act_pi = policy_fn(pi_nn, "pi", ob_space, ac_space, 1, reuse = True)
        OLDLOGPAC = tf.placeholder(tf.float32, [None])
        OLDVPRED = tf.placeholder(tf.float32, [None])

    else:
        pi = policy_fn(pi_nn, "pi", ob_space, ac_space, optim_batchsize, reuse=(pretrained_weight != None))  # Construct network for new policy
        oldpi = policy_fn(pi_nn, "oldpi", ob_space, ac_space, optim_batchsize) # Network for old policy
        ob = U.get_placeholder_cached(name="ob")

    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedul
    ac = pi.pdtype.sample_placeholder([None])

    if pi_nn == "gru" or pi_nn == "lstm":
        ratio = tf.exp(pi.pd.logp(ac) - OLDLOGPAC)

    else:
        ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold

    ent = pi.pd.entropy()
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent

    surr1 = ratio * atarg # surrogate from conservative policy iteration #ratio
    temp = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param)
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)

    if pi_nn == "gru" or pi_nn == "lstm":
        vpredclipped = OLDVPRED + tf.clip_by_value(pi.vpred - OLDVPRED, -clip_param, clip_param)
        vf_losses1 = tf.square(pi.vpred - ret)
        vf_losses2 = tf.square(vpredclipped - ret)
        vf_loss = tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
    else:
        vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))

    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "ent"]

    var_list = pi.get_trainable_variables()

    adam = MpiAdam(var_list, epsilon=adam_epsilon)
    d_adam = MpiAdam(reward_giver.get_trainable_variables())

    if pi_nn == "gru" or pi_nn == "lstm":
        compute_losses = U.function([pi.ob, ac, atarg, ret, lrmult, pi.S, pi.M, OLDLOGPAC, OLDVPRED], losses)
        lossandgrad = U.function([pi.ob, ac, atarg, ret, lrmult, pi.S, pi.M, OLDLOGPAC,OLDVPRED], losses + [U.flatgrad(total_loss, var_list, max_grad_norm)])

    else:
        assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv) for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
        compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)
        lossandgrad = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])

    U.initialize()
    adam.sync()
    d_adam.sync()

    # Prepare for rollouts
    # ----------------------------------------
    if pi_nn == "gru" or pi_nn == "lstm":
        runner = Runner(env=env, model=act_pi, nsteps=timesteps_per_actorbatch, gamma=gamma, lam=lam, reward_giver=reward_giver)
    else:
        seg_gen = traj_segment_generator(pi, env, reward_giver, timesteps_per_actorbatch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0####
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    true_rewbuffer = deque(maxlen=100)#40 or 100
    epinfo_buffer = deque(maxlen=100)

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"

    writer = tf.summary.FileWriter(log_dir + "/ppo")

    g_loss_stats = stats(loss_names)
    d_loss_stats = stats(reward_giver.loss_name)
    ep_stats = stats(["True_rewards", "Episode_length"])

    if load_model_path is not None:
        print(load_model_path)
        U.load_state(load_model_path)

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        if rank == 0 and iters_so_far % save_per_iter == 0 and ckpt_dir is not None:
            fname = os.path.join(ckpt_dir, task_name)
            fname += str(iters_so_far)
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            saver = tf.train.Saver()
            saver.save(tf.get_default_session(), fname)
        logger.log("********** Iteration %i ************"%iters_so_far)

        logger.log("Optimizing Policy...")
        with timed("sampling", rank):
            if pi_nn == "mlp":
                seg = seg_gen.__next__()
                add_vtarg_and_adv(seg, gamma, lam)
                ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
                atarg = (atarg - atarg.mean()) / atarg.std()
                vpredbefore = seg["vpred"]
                d = dataset.Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret),
                    deterministic=pi.recurrent)

            else:
                ob, tdlamret, masks, ac, values, logps, state, epinfos = runner.run()
                atarg = tdlamret - values
                epinfo_buffer.extend(epinfos)
                atarg = (atarg - atarg.mean()) / (atarg.std() + 1e-8)
                vpredbefore = values
                d = dataset.Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret, masks=masks, logps=logps, values=values),
                    deterministic=pi.recurrent)

        optim_batchsize = optim_batchsize or ob.shape[0]
        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        if pi_nn == "mlp":
            assign_old_eq_new() # set old parameter values to new parameter values
        logger.log("Optimizing...")

        # Here we do a bunch of optimization epochs over the data
        if pi_nn == "mlp":
            for _ in range(optim_epochs):
                losses = [] # list of tuples, each of which gives the loss for a minibatch
                for batch in d.iterate_once(optim_batchsize):
                    with timed("computegrad", rank):
                        *newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                    adam.update(g, optim_stepsize * cur_lrmult)
                    losses.append(newlosses)
                logger.log(fmt_row(13, loss_names))
                logger.log(fmt_row(13, np.mean(losses, axis=0)))
        else:
            for _ in range(optim_epochs):
                losses = []
                batches = []
                for batch in d.iterate_once(optim_batchsize):
                    batches.append(batch)
                inds = np.arange(len(batches))
                np.random.shuffle(inds)
                for i in range(len(batches)):
                    with timed("computegrad", rank):
                        ind = inds[i]
                        *newlosses, g = lossandgrad(batches[ind]["ob"], batches[ind]["ac"], batches[ind]["atarg"], batches[ind]["vtarg"],
                                                    cur_lrmult, state, batches[ind]["masks"], batches[ind]["logps"], batches[ind]["values"])
                    adam.update(g, optim_stepsize * cur_lrmult)
                    losses.append(newlosses)
                logger.log(fmt_row(13, loss_names))
                logger.log(fmt_row(13, np.mean(losses, axis=0)))

        logger.log("Evaluating losses...")
        losses = []
        for batch in d.iterate_once(optim_batchsize):
            if pi_nn == "gru" or pi_nn == "lstm":
                newlosses = compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"],
                                           cur_lrmult, state, batch["masks"], batch["logps"], batch["values"])

            else:
                newlosses = compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
            losses.append(newlosses)
        meanlosses,_,_ = mpi_moments(losses, axis=0)
        logger.log(fmt_row(13, meanlosses))
        for (lossval, name) in zipsame(meanlosses, loss_names):
            logger.record_tabular("loss_"+name, lossval)
        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
        g_losses = meanlosses

        # ------------------ Update D ------------------
        logger.log("Optimizing Discriminator...")
        logger.log(fmt_row(13, reward_giver.loss_name))
        ob_expert, ac_expert = expert_dataset.get_next_batch(len(ob))
        batch_size = len(ob) // d_step
        d_losses = []  # list of tuples, each of which gives the loss for a minibatch
        for ob_batch, ac_batch in dataset.iterbatches((ob, ac),
                                                      include_final_partial_batch=False,
                                                      batch_size=batch_size):
            ob_expert, ac_expert = expert_dataset.get_next_batch(len(ob_batch))
            # update running mean/std for reward_giver
            if hasattr(reward_giver, "obs_rms"): reward_giver.obs_rms.update(np.concatenate((ob_batch, ob_expert), 0))
            *newlosses, g = reward_giver.lossandgrad(ob_batch, ac_batch, ob_expert, ac_expert)
            d_adam.update(allmean(g, nworkers), d_stepsize)
            d_losses.append(newlosses)
        logger.log(fmt_row(13, np.mean(d_losses, axis=0)))

        if pi_nn == "mlp":
            lrlocal = (seg["ep_lens"], seg["ep_true_rets"]) # local values
            listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
            lens, true_rets = map(flatten_lists, zip(*listoflrpairs))
            true_rewbuffer.extend(true_rets)
            lenbuffer.extend(lens)

            logger.record_tabular("EpLenMean", np.mean(lenbuffer))
            logger.record_tabular("EpTrueRewMean", np.mean(true_rewbuffer))
            logger.record_tabular("EpThisIter", len(lens))
            episodes_so_far += len(lens)
            timesteps_so_far += sum(lens)
            iters_so_far += 1
            logger.record_tabular("EpisodesSoFar", episodes_so_far)
            logger.record_tabular("TimestepsSoFar", timesteps_so_far)
            logger.record_tabular("TimeElapsed", time.time() - tstart)

        else:
            lens = [epinfo['l'] for epinfo in epinfo_buffer]
            true_rets = [epinfo['r'] for epinfo in epinfo_buffer]

            logger.record_tabular("EpLenMean", safemean(lens))
            logger.record_tabular("EpTrueRewMean", safemean(true_rets))

            timesteps_so_far += timesteps_per_actorbatch
            iters_so_far += 1

            logger.record_tabular("ItersSoFar", iters_so_far)
            logger.record_tabular("TimestepsSoFar", timesteps_so_far)
            logger.record_tabular("TimeElapsed", time.time() - tstart)

        if rank == 0:
            logger.dump_tabular()
            g_loss_stats.add_all_summary(writer, g_losses, iters_so_far)
            d_loss_stats.add_all_summary(writer, np.mean(d_losses, axis=0), iters_so_far)

            if pi_nn == "mlp":
                ep_stats.add_all_summary(writer, [np.mean(true_rewbuffer), np.mean(lenbuffer)], iters_so_far)
            else:
                ep_stats.add_all_summary(writer, [safemean(true_rets), safemean(lens)], iters_so_far)
    return pi

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)
