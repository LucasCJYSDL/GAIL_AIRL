import time
import os

from mpi4py import MPI
from collections import deque

import tensorflow as tf
import numpy as np

import baselines.common.tf_util as U
from baselines.common import explained_variance, zipsame, dataset, fmt_row
from baselines import logger

from baselines.common.mpi_adam import MpiAdam
from baselines.common.cg import cg
from baselines.gail.statistics import stats

from train_util import allmean, timed, traj_segment_generator, add_vtarg_and_adv


def learn(env, policy_func, expert_dataset, rank, pretrained, pretrained_weight, *,
          entcoeff, save_per_iter, ckpt_dir, log_dir, timesteps_per_batch, task_name,
          gamma, lam, max_kl, cg_iters, cg_damping=1e-2,
          vf_stepsize=3e-4, vf_iters=3,
          max_timesteps=0, max_episodes=0, max_iters=0,
          callback=None, load_path = None, use_true_reward = False, use_sparse_reward = False):

    nworkers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    np.set_printoptions(precision=3)
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space, reuse=(pretrained_weight != None))
    oldpi = policy_func("oldpi", ob_space, ac_space)
    atarg = tf.placeholder(dtype=tf.float32, shape=[None])  # Target advantage function (if applicable)
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
    compute_losses = U.function([ob, ac, atarg], losses)
    compute_lossandgrad = U.function([ob, ac, atarg], losses + [U.flatgrad(optimgain, var_list)])
    compute_fvp = U.function([flat_tangent, ob, ac, atarg], fvp)
    compute_vflossandgrad = U.function([ob, ret], U.flatgrad(vferr, vf_var_list))


    U.initialize()
    th_init = get_flat()
    MPI.COMM_WORLD.Bcast(th_init, root=0)
    set_from_flat(th_init)
    vfadam.sync()

    if rank == 0:
        print("Init param sum", th_init.sum(), flush=True)

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_batch, stochastic=True)#Iterater

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=40)  # rolling buffer for episode lengths
    true_rewbuffer = deque(maxlen=40)
    new_lenbuffer = deque(maxlen=40)  # decision making steps
    new_true_rewbuffer = deque(maxlen=40) # no matter whether the episode is done or not

    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0]) == 1

    writer = tf.summary.FileWriter(log_dir + "/trpo")

    g_loss_stats = stats(loss_names)
    ep_stats = stats(["True_rewards","Changing_length", "Done_ratio", "Accumulated_rewards", "Decision_length"])
    # if provide pretrained weight

    if pretrained_weight is not None:
        U.load_variables(pretrained_weight, variables=pi.get_variables())

    if load_path is not None:
        U.load_state(load_path)

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


        # ------------------ Sampling------------------
        with timed("sampling", rank):
            seg = seg_gen.__next__()

        ob, ac, mean, std = seg["ob"], seg["ac"], seg["mean"], seg["logstd"]


        # ------------------ get Reward and Advantage ------------------

        true_rew = np.expand_dims(seg["true_reward"], axis=1)
        sparse_rew = np.expand_dims(seg["sparse_reward"], axis=1)


        if use_true_reward:
            seg["rew"] = rewards = true_rew ##reward_2

            # print(sum(true_rew))

        elif use_sparse_reward:
            seg["rew"] = rewards = sparse_rew  ##reward_2

        add_vtarg_and_adv(seg, gamma, lam)

        # ------------------ Update G ------------------
        def fisher_vector_product(p):
            return allmean(compute_fvp(p, *fvpargs), nworkers) + cg_damping * p

        logger.log("Optimizing Policy...")
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"]  # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob)  # update running mean/std for policy

        args = seg["ob"], seg["ac"], atarg
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
                for (mbob, mbret) in dataset.iterbatches((seg["ob"], seg["tdlamret"]),
                                                         include_final_partial_batch=False, batch_size=128):
                    if hasattr(pi, "ob_rms"):
                        pi.ob_rms.update(mbob)  # update running mean/std for policy
                    g = allmean(compute_vflossandgrad(mbob, mbret), nworkers)
                    vfadam.update(g, vf_stepsize)

        g_losses = meanlosses
        for (lossname, lossval) in zip(loss_names, meanlosses):
            logger.record_tabular(lossname, lossval)
        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))

        # ------------------ Log info ------------------

        lrlocal = (seg["ep_lens"], seg["ep_true_rets"], seg["new_lens"], seg["new_true_rets"])  # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
        lens, true_rets, new_lens, new_true_rets = map(flatten_lists, zip(*listoflrpairs))
        true_rewbuffer.clear()
        lenbuffer.clear()
        new_true_rewbuffer.clear()
        new_lenbuffer.clear()
        true_rewbuffer.extend(true_rets)
        lenbuffer.extend(lens)
        new_true_rewbuffer.extend(new_true_rets)
        new_lenbuffer.extend(new_lens)

        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpTrueRewMean", np.mean(true_rewbuffer))
        logger.record_tabular("NewLenMean", np.mean(new_lenbuffer))
        logger.record_tabular("NewTrueRewMean", np.mean(new_true_rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        logger.record_tabular("Done_ratio", seg["done_ratio"])
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1
        if iters_so_far >15300:
            break

        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)

        if rank == 0:

            logger.dump_tabular()
            g_loss_stats.add_all_summary(writer, g_losses, iters_so_far)
            ep_stats.add_all_summary(writer, [np.mean(true_rewbuffer), np.mean(lenbuffer), seg["done_ratio"], np.mean(new_true_rewbuffer), np.mean(new_lenbuffer)], iters_so_far)


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]