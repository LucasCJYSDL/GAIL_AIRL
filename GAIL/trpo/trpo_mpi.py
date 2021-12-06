'''
Disclaimer: The trpo part highly rely on trpo_mpi at @openai/baselines
'''

import time
import os

from mpi4py import MPI
from collections import deque

import tensorflow as tf
import numpy as np

import common.tf_util as U
from baselines.common import explained_variance, zipsame, dataset, fmt_row
from baselines import logger

from baselines.common.mpi_adam import MpiAdam
from baselines.common.cg import cg
from common_util.statistics import stats
from common_util.train_util import traj_segment_generator, add_vtarg_and_adv, timed, allmean

from baselines.common.distributions import make_pdtype
from baselines.common.mpi_moments import mpi_moments


def learn(env, policy_func, reward_giver, expert_dataset, rank, pi_nn,
          pretrained, pretrained_weight, *,
          g_step, d_step, entcoeff, save_per_iter, optim_batchsize,
          ckpt_dir, log_dir, timesteps_per_batch, task_name,
          gamma, lam, max_grad_norm,
          max_kl, cg_iters, cg_damping=1e-2,
          vf_stepsize=3e-4, d_stepsize=3e-4, vf_iters=3,
          max_timesteps=0, max_episodes=0, max_iters=0,
          callback=None, schedule = 'linear', load_model_path = None
          ):

    nworkers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    np.set_printoptions(precision=3)
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space

    if pi_nn == "mlpgru":
        pi = policy_func(pi_nn, "pi", ob_space, ac_space, optim_batchsize, reuse=(pretrained_weight != None)) # Construct network for new policy
        act_pi = policy_func(pi_nn, "pi", ob_space, ac_space, 1, reuse = True)

        OLDLOGPAC = tf.placeholder(tf.float32, [None])
        OLDPDPARAM = tf.placeholder(tf.float32, [None, 6])

    else:
        pi = policy_func(pi_nn, "pi", ob_space, ac_space, optim_batchsize, reuse=(pretrained_weight != None))  # ##
        oldpi = policy_func(pi_nn, "oldpi", ob_space, ac_space, optim_batchsize)                               #pi的输出与oldpi的输出不应偏差太大
        ob = U.get_placeholder_cached(name="ob")
        old_ac = pi.pdtype.sample_placeholder([None])


    atarg = tf.placeholder(dtype=tf.float32, shape=[None])  # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return
    ac = pi.pdtype.sample_placeholder([None])


    if pi_nn == "mlpgru":
        pdtype = make_pdtype(ac_space)
        opd = pdtype.pdfromflat(OLDPDPARAM)
        kloldnew = opd.kl(pi.pd)
        ratio = tf.exp(pi.pd.logp(ac) - OLDLOGPAC)

    else:
        kloldnew = oldpi.pd.kl(pi.pd)
        ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))

    ent = pi.pd.entropy()###8 公式是啥
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    entbonus = entcoeff * meanent

    vferr = tf.reduce_mean(tf.square(pi.vpred - ret))#ret即为累计误差
    surrgain = tf.reduce_mean(ratio * atarg)

    optimgain = surrgain + entbonus
    losses = [optimgain, meankl, entbonus, surrgain, meanent]
    loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]

    dist = meankl
    all_var_list = pi.get_trainable_variables()

    var_list = [v for v in all_var_list if v.name.startswith("pi/pol") or v.name.startswith("pi/logstd")]
    vf_var_list = [v for v in all_var_list if v.name.startswith("pi/vff")]


    d_adam = MpiAdam(reward_giver.get_trainable_variables())
    vfadam = MpiAdam(vf_var_list)

    get_flat = U.GetFlat(var_list)
    set_from_flat = U.SetFromFlat(var_list)
    klgrads = tf.gradients(dist, var_list)
    flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
    shapes = [var.get_shape().as_list() for var in var_list]
    start = 0
    tangents = []
    for shape in shapes:
        sz = U.intprod(shape)
        tangents.append(tf.reshape(flat_tangent[start:start+sz], shape))
        start += sz
    gvp = tf.add_n([tf.reduce_sum(g*tangent) for (g, tangent) in zipsame(klgrads, tangents)])  # pylint: disable=E1111
    if pi_nn == "mlpgru":
        fvp = U.flatgrad(gvp, var_list, max_grad_norm)
    else:
        fvp = U.flatgrad(gvp, var_list)
        assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
                                                    for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])

    if pi_nn == 'mlpgru':
        compute_losses = U.function([pi.ob, ac, atarg, pi.S, pi.M, OLDLOGPAC, OLDPDPARAM], losses)
        compute_lossandgrad = U.function([pi.ob, ac, atarg, pi.S, pi.M, OLDLOGPAC, OLDPDPARAM], losses + [U.flatgrad(optimgain, var_list)])
        compute_fvp = U.function([flat_tangent, pi.ob, ac, atarg, pi.S, pi.M, OLDLOGPAC, OLDPDPARAM], fvp)
        compute_vflossandgrad = U.function([pi.ob, ret], U.flatgrad(vferr, vf_var_list))  # ？？？ret是啥
    else:
        compute_losses = U.function([ob, ac, atarg], losses)
        compute_lossandgrad = U.function([ob, ac, atarg], losses + [U.flatgrad(optimgain, var_list)])
        compute_fvp = U.function([flat_tangent, ob, ac, atarg], fvp)
        compute_vflossandgrad = U.function([ob, ret], U.flatgrad(vferr, vf_var_list))#？？？ret是啥

    U.initialize()
    th_init = get_flat()
    MPI.COMM_WORLD.Bcast(th_init, root=0)
    set_from_flat(th_init)
    d_adam.sync()
    vfadam.sync()

    if rank == 0:
        print("Init param sum", th_init.sum(), flush=True)

    # Prepare for rollouts
    # ----------------------------------------
    if pi_nn == 'mlpgru':
        from common_util.runner import Runner
        runner = Runner(env=env, model=act_pi, pi_nn = pi_nn, nsteps=timesteps_per_batch, gamma=gamma, lam=lam, reward_giver=reward_giver)
    else:
        seg_gen = traj_segment_generator(pi, env, reward_giver, timesteps_per_batch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()

    epinfo_buffer = deque(maxlen=100)
    lenbuffer = deque(maxlen=100)  # rolling buffer for episode lengths
    true_rewbuffer = deque(maxlen=100)

    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0]) == 1

    writer = tf.summary.FileWriter(log_dir+"/trpo")

    g_loss_stats = stats(loss_names)
    d_loss_stats = stats(reward_giver.loss_name)
    ep_stats = stats(["True_rewards","Episode_length"])

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

        # Save model
        if rank == 0 and iters_so_far % save_per_iter == 0 and ckpt_dir is not None:
            fname = os.path.join(ckpt_dir, task_name)
            fname += str(iters_so_far)
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            saver = tf.train.Saver()
            saver.save(tf.get_default_session(), fname)

        logger.log("********** Iteration %i ************" % iters_so_far)

        # ------------------ Update G ------------------

        logger.log("Optimizing Policy...")
        losses = []

        def fisher_vector_product(p):
            return allmean(compute_fvp(p, *fvpargs), nworkers) + cg_damping * p

        def train_step_G(args, enlarger):
            with timed("computegrad", rank):
                *lossbefore, g = compute_lossandgrad(*args)

            if pi_nn == "mlp":
                assign_old_eq_new()

            lossbefore = allmean(np.array(lossbefore), nworkers)
            g = allmean(g, nworkers)
            if np.allclose(g, 0):
                logger.log("Got zero gradient. not updating")
            else:
                with timed("cg", rank):
                    stepdir = cg(fisher_vector_product, g, cg_iters=cg_iters, verbose=rank == 0)
                assert np.isfinite(stepdir).all()
                shs = .5 * stepdir.dot(fisher_vector_product(stepdir))
                lm = np.sqrt(shs / max_kl)
                fullstep = stepdir / lm
                expectedimprove = g.dot(fullstep)
                surrbefore = lossbefore[0]
                stepsize = 1.0
                thbefore = get_flat()
                for _ in range(10):
                    thnew = thbefore + fullstep * stepsize
                    set_from_flat(thnew)
                    meanlosses = surr, kl, *_ = allmean(np.array(compute_losses(*args)), nworkers)
                    improve = surr - surrbefore
                    logger.log("Expected: %.3f Actual: %.3f" % (expectedimprove, improve))
                    if not np.isfinite(meanlosses).all():
                        logger.log("Got non-finite value of losses -- bad!")
                    elif kl > max_kl * enlarger:
                        logger.log("violated KL constraint. shrinking step.")
                    elif improve < 0:
                        logger.log("surrogate didn't improve. shrinking step.")
                    else:
                        logger.log("Stepsize OK!")
                        break
                    stepsize *= .5
                else:
                    logger.log("couldn't compute a good step")
                    set_from_flat(thbefore)
                losses.append(meanlosses)
                if nworkers > 1 and iters_so_far % 20 == 0:
                    paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), vfadam.getflat().sum()))  # list of tuples
                    assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])
                return meanlosses

        for i in range(g_step):

            with timed("sampling", rank):
                if pi_nn == "mlpgru":
                    ob, tdlamret, masks, ac, values, logps, pdparams, state, epinfos = runner.run()
                    atarg = tdlamret - values
                    vpredbefore = values
                    if i == 0:
                        epinfo_buffer.extend(epinfos)
                    atarg = (atarg - atarg.mean()) / (atarg.std() + 1e-8)

                else:
                    seg = seg_gen.__next__()
                    add_vtarg_and_adv(seg, gamma, lam)
                    ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
                    vpredbefore = seg["vpred"]
                    atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate

            if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob)  # update running mean/std for policy###2

            if pi_nn == "mlpgru":

                d = dataset.Dataset(
                    dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret, masks=masks, logps=logps, pdparam=pdparams),
                    deterministic=pi.recurrent)
                optim_batchsize = optim_batchsize or ob.shape[0]

                batches = []
                for batch in d.iterate_once(optim_batchsize):
                    batches.append(batch)
                inds = np.arange(len(batches))
                np.random.shuffle(inds)
                for j in range(timesteps_per_batch//optim_batchsize):
                    ind = inds[j]
                    args = batches[ind]["ob"], batches[ind]["ac"], batches[ind]["atarg"], \
                           state, batches[ind]["masks"], batches[ind]["logps"], batches[ind]["pdparam"]
                    fvpargs = [arr[:] for arr in args]
                    train_step_G(args, 2.0)

            else:

                args = seg["ob"], seg["ac"], atarg
                fvpargs = [arr[::5] for arr in args]
                meanloss = train_step_G(args, 1.5)

            with timed("vf", rank):
                for _ in range(vf_iters):
                    for (mbob, mbret) in dataset.iterbatches((ob, tdlamret),
                                                             include_final_partial_batch=False, batch_size=optim_batchsize):##
                        if hasattr(pi, "ob_rms"):
                            pi.ob_rms.update(mbob)  # update running mean/std for policy ###2
                        g = allmean(compute_vflossandgrad(mbob, mbret), nworkers)
                        vfadam.update(g, vf_stepsize)

        if pi_nn == "mlpgru":
            meanloss, _, _ = mpi_moments(losses, axis=0)###???
        g_losses = meanloss
        for (lossname, lossval) in zip(loss_names, meanloss):
            logger.record_tabular(lossname, lossval)
        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
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
            if hasattr(reward_giver, "obs_rms"): reward_giver.obs_rms.update(np.concatenate((ob_batch, ob_expert), 0))###2
            *newlosses, g = reward_giver.lossandgrad(ob_batch, ac_batch, ob_expert, ac_expert)
            d_adam.update(allmean(g, nworkers), d_stepsize)
            d_losses.append(newlosses)
        logger.log(fmt_row(13, np.mean(d_losses, axis=0)))

        if pi_nn == "mlpgru":
            lens = [epinfo['l'] for epinfo in epinfo_buffer]
            true_rets = [epinfo['r'] for epinfo in epinfo_buffer]

            logger.record_tabular("EpLenMean", safemean(lens))
            logger.record_tabular("EpTrueRewMean", safemean(true_rets))

            timesteps_so_far += timesteps_per_batch
            iters_so_far += 1

            logger.record_tabular("ItersSoFar", iters_so_far)
            logger.record_tabular("TimestepsSoFar", timesteps_so_far)
            logger.record_tabular("TimeElapsed", time.time() - tstart)

        else:
            lrlocal = (seg["ep_lens"], seg["ep_true_rets"])  # local values
            listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
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

        if rank == 0:
            logger.dump_tabular()
            g_loss_stats.add_all_summary(writer, g_losses, iters_so_far)
            d_loss_stats.add_all_summary(writer, np.mean(d_losses, axis=0), iters_so_far)
            if pi_nn == "mlpgru":
                ep_stats.add_all_summary(writer, [safemean(true_rets), safemean(lens)], iters_so_far)
            else:
                ep_stats.add_all_summary(writer, [np.mean(true_rewbuffer), np.mean(lenbuffer)], iters_so_far)


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)



