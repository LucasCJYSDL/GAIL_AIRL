import os

from mpi4py import MPI

import tensorflow as tf
import numpy as np

import baselines.common.tf_util as U
from baselines.common import explained_variance, zipsame, dataset, fmt_row
from baselines import logger

from baselines.common.mpi_adam import MpiAdam
from baselines.common.cg import cg
from baselines.gail.statistics import stats

from replay_buffer import ReplayBuffer

from train_util import *


def learn(env, policy_func, discriminator, expert_dataset, rank, *,
          posterior, posterior_target,
          d_step, d_iters, p_iters, entcoeff, save_per_iter,
          ckpt_dir, log_dir, task_name,
          gamma, lam, max_kl, cg_iters, cg_damping=1e-2,
          vf_stepsize=3e-4, d_stepsize=3e-5, p_stepsize=3e-5, vf_iters=5,
          max_timesteps=0, max_episodes=0, max_iters=0, buffer_size=40, sample_size=20, coe_d=1, coe_p=1.5, coe_t=1,
          encode_num=3, callback=None, load_path = None, use_true_reward = False, use_sparse_reward = False
          ):

    nworkers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    np.set_printoptions(precision=3)
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space, encode_num=encode_num, reuse=False)
    oldpi = policy_func("oldpi", ob_space, ac_space, encode_num=encode_num)
    atarg = tf.placeholder(dtype=tf.float32, shape=[None])  # Target advantage function (if applicable)
    encode_ph = U.get_placeholder_cached(name="encode_ph")
    ret = tf.placeholder(dtype=tf.float32, shape=[None])

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)

    meanent = tf.stop_gradient(meanent)

    entbonus = entcoeff * meanent

    vferr = tf.reduce_mean(tf.square(pi.vpred - ret))

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))  # advantage * pnew / pold
    surrgain = tf.reduce_mean(ratio * atarg)

    optimgain = surrgain + entbonus
    losses = [optimgain, meankl, entbonus, surrgain, meanent]
    loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]

    dist = meankl

    all_var_list = pi.get_trainable_variables()
    var_list = [v for v in all_var_list if v.name.startswith("pi/pol") or v.name.startswith("pi/logstd")]
    vf_var_list = [v for v in all_var_list if v.name.startswith("pi/vff")]

    d_adam = MpiAdam(discriminator.get_trainable_variables())
    p_adam = MpiAdam(posterior.get_trainable_variables())
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
    fvp = U.flatgrad(gvp, var_list)

    assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
                                                    for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])

    assign_poster_eq_target = U.function([], [], updates=[tf.assign(oldv, 0.5*newv+0.5*oldv)
                                                    for (oldv, newv) in zipsame(posterior_target.get_variables(), posterior.get_variables())])

    compute_losses = U.function([ob, ac, atarg, encode_ph], losses)
    compute_lossandgrad = U.function([ob, ac, atarg, encode_ph], losses + [U.flatgrad(optimgain, var_list)])
    compute_fvp = U.function([flat_tangent, ob, ac, atarg, encode_ph], fvp)
    compute_vflossandgrad = U.function([ob, ret, encode_ph], U.flatgrad(vferr, vf_var_list))


    U.initialize()
    th_init = get_flat()
    MPI.COMM_WORLD.Bcast(th_init, root=0)
    set_from_flat(th_init)
    d_adam.sync()
    p_adam.sync()
    vfadam.sync()

    if rank == 0:
        print("Init param sum", th_init.sum(), flush=True)

    # Prepare for rollouts
    # ----------------------------------------
    #seg_gen = traj_segment_generator(pi, env, path_num, stochastic=True)#Iterater
    buffer = ReplayBuffer(buffer_size)
    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()


    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0]) == 1

    writer = [tf.summary.FileWriter(log_dir + "/trpo/style" + str(i)) for i in range(encode_num+1)]

    metrics_name = ["True_rewards", "Post_rewards", "Disc_rewards", "Total_rewards", "Run_step", "Decision_step", "Change_step",\
                      "Max_acc", "Min_acc", "Mp_acc", "Mn_acc", "Vp_acc", "Vn_acc",\
                      "Max_spd", "Min_spd", "M_spd", "V_spd", "Num_pos_time", "Num_neg_time",\
                      "Gap_dis_lead", "Gap_dis_tail", "Gap_dis_min", "Gap_ind", "Done_ratio"]

    g_loss_stats = stats(loss_names)
    d_loss_stats = stats(discriminator.loss_name)
    p_loss_stats = stats(posterior.loss_name)
    ep_stats = stats(metrics_name)

    # if provide pretrained weight

    if load_path is not None:
        U.load_state(load_path)

    prev_eval_metrics = [np.array([0]*len(metrics_name)) for _ in range(encode_num)]

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

        if iters_so_far == 0:
            path_num = 20
        else:
            path_num = 10

        # ------------------ Sampling------------------
        with timed("sampling", rank):
            segs, evals = traj_segment_generator(pi, env, path_num, posterior_target, stochastic=True, encode_num=encode_num)#Iterater

            for i in range(len(evals)):
                evals[i]["metrics"] = get_eval_metrics(evals[i]["metrics"])

        for seg in segs:
            seg["lprobs"] = compute_path_probs(seg["ac"], seg["mean"], seg["logstd"], ac_space, is_expert=False)
            buffer.add(seg)

        segs = buffer.get_sample(sample_size)
        ob = np.concatenate([seg["ob"] for seg in segs])
        ac = np.concatenate([seg["ac"] for seg in segs])
        mean = np.concatenate([seg["mean"] for seg in segs])
        std = np.concatenate([seg["logstd"] for seg in segs])
        lprobs = np.concatenate([seg["lprobs"] for seg in segs])
        encodes = np.concatenate([seg["encode"] for seg in segs])

        # ------------------ Update D ------------------
        logger.log("Optimizing Discriminator...")
        logger.log(fmt_row(13, discriminator.loss_name))
        batch_size = len(ob) // d_step

        labels = np.zeros((batch_size * 2, 1))
        labels[batch_size:] = 1.0

        d_losses = []  # list of tuples, each of which gives the loss for a minibatch
        for i in range(d_iters):
            for ob_batch, ac_batch, lprobs_batch in dataset.iterbatches((ob, ac, lprobs), shuffle=True,
                                                          include_final_partial_batch=False,
                                                          batch_size=batch_size):
                ob_expert, ac_expert, mean_expert, std_expert = expert_dataset.get_next_batch(len(ob_batch))
                lprobs_expert = compute_path_probs(ac_expert, mean_expert, std_expert, ac_space, is_expert=True)
                # update running mean/std for discriminator
                if hasattr(discriminator, "obs_rms"): discriminator.obs_rms.update(np.concatenate((ob_batch, ob_expert), 0))
                lprob_batch = np.concatenate([lprobs_batch, lprobs_expert], axis=0)

                *newlosses, g = discriminator.lossandgrad(ob_batch, ac_batch, ob_expert, ac_expert, labels, lprob_batch)
                d_adam.update(allmean(g, nworkers), d_stepsize)
                d_losses.append(newlosses)
        logger.log(fmt_row(13, np.mean(d_losses, axis=0)))

        # ------------------ update P ------------------
        logger.log("Optimizing Posterior...")
        logger.log(fmt_row(13, posterior.loss_name))

        p_losses = []
        for i in range(p_iters):
            for ob_batch, ac_batch, encodes_batch in dataset.iterbatches((ob, ac, encodes), shuffle=True,
                                                          include_final_partial_batch=False,
                                                          batch_size=batch_size):

                if hasattr(posterior, "obs_rms"): posterior.obs_rms.update(ob_batch)

                *newlosses, g = posterior.lossandgrad(ob_batch, ac_batch, encodes_batch)
                p_adam.update(allmean(g, nworkers), p_stepsize)

                #with timed("assign", rank):

                assign_poster_eq_target()

                p_losses.append(newlosses)

        logger.log(fmt_row(13, np.mean(p_losses, axis=0)))


        # ------------------ get Reward and Advantage ------------------

        Disc_rewards = []
        Total_rewards = []
        for i in range(encode_num):
            Disc_rewards.append([])
            Total_rewards.append([])

        for seg in segs:
            new_batch_size = seg["ob"].shape[0]
            lprobs_rew = seg["lprobs"]
            Disc_reward = discriminator.get_reward(seg["ob"], seg["ac"], new_batch_size, lprobs_rew)
            Post_reward = posterior_target.get_reward(seg["ob"], seg["ac"], seg["encode"])
            rewards =  coe_d * Disc_reward + coe_p * Post_reward

            if use_true_reward:
                true_rew = np.expand_dims(seg["true_reward"], axis=1)
                rewards +=  coe_t * true_rew ##reward_2

            elif use_sparse_reward:
                print('rewards: ', rewards)
                sparse_rew = np.expand_dims(seg["sparse_reward"], axis=1)
                print("sparse: ", sparse_rew)
                rewards += coe_t * sparse_rew  ##reward_2

            seg["rew"] = rewards

            ind = np.argmax(seg["encode"])
            Disc_rewards[ind].append(np.sum(Disc_reward))
            Total_rewards[ind].append(np.sum(rewards))
            add_vtarg_and_adv(seg, gamma, lam)

        # ------------------ Update G ------------------
        def fisher_vector_product(p):
            return allmean(compute_fvp(p, *fvpargs), nworkers) + cg_damping * p

        logger.log("Optimizing Policy...")
        atarg = np.concatenate([seg["adv"] for seg in segs])
        tdlamret = np.concatenate([seg["tdlamret"] for seg in segs])
        vpredbefore = np.concatenate([seg["vpred"] for seg in segs])

        atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob)  # update running mean/std for policy

        args = ob, ac, atarg, encodes
        fvpargs = [arr[::5] for arr in args]

        with timed("computegrad", rank):
            *lossbefore, g = compute_lossandgrad(*args)

        assign_old_eq_new()  # set old parameter values to new parameter values
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
            # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
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
                elif kl > max_kl * 1.5:
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
            if nworkers > 1 and iters_so_far % 20 == 0:
                paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), vfadam.getflat().sum()))  # list of tuples
                assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])
        with timed("vf", rank):
            for _ in range(vf_iters):
                for (mbob, mbret, mbend) in dataset.iterbatches((ob, tdlamret, encodes), shuffle=True,
                                                         include_final_partial_batch=False, batch_size=128):
                    if hasattr(pi, "ob_rms"):
                        pi.ob_rms.update(mbob)  # update running mean/std for policy
                    g = allmean(compute_vflossandgrad(mbob, mbret, mbend), nworkers)
                    vfadam.update(g, vf_stepsize)

        g_losses = meanlosses
        for (lossname, lossval) in zip(loss_names, meanlosses):
            logger.record_tabular(lossname, lossval)
        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))

        # ------------------ Log info ------------------

        logger.record_tabular("EpThisIter", path_num)
        iters_so_far += 1
        if iters_so_far >45300:
            break
        logger.record_tabular("TimeElapsed", time.time() - tstart)

        if rank == 0:
            logger.dump_tabular()
            g_loss_stats.add_all_summary(writer[encode_num], g_losses, iters_so_far)
            d_loss_stats.add_all_summary(writer[encode_num], np.mean(d_losses, axis=0), iters_so_far)
            p_loss_stats.add_all_summary(writer[encode_num], np.mean(p_losses, axis=0), iters_so_far)

        for i in range(encode_num):
            temp = evals[i]
            metric = temp["metrics"]
            step_num = extract_num(metric["gaps"]["gap_idx_hist"])

            lrlocal = [temp["ep_rets"], temp["post_rets"], Disc_rewards[i], Total_rewards[i], metric["steps"]["run_step"], metric["steps"]["decision_step"], metric["steps"]["change_step"],\
                       metric["acces"]["max"], metric["acces"]["min"],metric["acces"]["mean_pos"],metric["acces"]["mean_neg"],metric["acces"]["var_pos"],metric["acces"]["var_neg"],\
                       metric["speeds"]["max"], metric["speeds"]["min"], metric["speeds"]["mean"], metric["speeds"]["var"],metric["times"]["num_pos_acce"],metric["times"]["num_neg_acce"],\
                       metric["gaps"]["dis_lead"],np.array(metric["gaps"]["dis_tail"]),np.array(metric["gaps"]["dis_min"])] # local values
            lrlocal.append(step_num)
            lrlocal.append([temp["done_ratio"]])

            #sliding with last evaluation metrics
            eval_metrics=np.array([np.mean(elem) for elem in lrlocal])
            eval_show = list(0.5*prev_eval_metrics[i] + 0.5*eval_metrics)
            prev_eval_metrics[i] = eval_metrics

            #print([np.mean(elem) for elem in lrlocal])
            if rank == 0:
                ep_stats.add_all_summary(writer[i], eval_show, iters_so_far)