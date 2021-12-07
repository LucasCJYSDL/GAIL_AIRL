import os
from mpi4py import MPI
import tensorflow as tf
from baselines.common import explained_variance, fmt_row, zipsame, dataset
from baselines import logger
from baselines.gail.statistics import stats
from train_util import *
from MetaLearner import MetaLearner
from run import runner
from baselines.common import tf_util as U
from baselines.common.mpi_adam import MpiAdam
from variables import VariableState, average_vars, interpolate_vars
from baselines.common.cg import cg


def learn(env, policy_func, discriminator, expert_dataset, rank, *,
          d_iters, entcoeff, save_per_iter, ckpt_dir, log_dir, task_name,
          gamma, lam, max_kl, cg_iters, cg_damping=1e-2,
          vf_stepsize=3e-4, d_stepsize=3e-5, vf_iters=5,
          max_timesteps=0, max_episodes=0, max_iters=0, coe_d=1, coe_t=1,
          style_num=2, path_num=10, update_times=4, eval_update_times=4, meta_stepsize=0.25, callback=None, load_path = None, use_true_reward = False, use_sparse_reward = False
          ):

    nworkers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    np.set_printoptions(precision=3)
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space, reuse=False)
    oldpi = policy_func("oldpi", ob_space, ac_space)


    # mlearner = MetaLearner(env, nworkers, rank, pi, oldpi, discriminator, entcoeff)
    #
    # mlearner.initialize()

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

    dis_var_list = discriminator.get_trainable_variables()
    pi_para_model = VariableState(all_var_list)
    dis_para_model = VariableState(dis_var_list)

    d_adam = MpiAdam(discriminator.get_trainable_variables())
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
        tangents.append(tf.reshape(flat_tangent[start:start + sz], shape))
        start += sz
    gvp = tf.add_n([tf.reduce_sum(g * tangent) for (g, tangent) in zipsame(klgrads, tangents)])  # pylint: disable=E1111
    fvp = U.flatgrad(gvp, var_list)

    assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
                                                    for (oldv, newv) in
                                                    zipsame(oldpi.get_variables(), pi.get_variables())])


    compute_losses = U.function([ob, ac, atarg], losses)
    compute_lossandgrad = U.function([ob, ac, atarg], losses + [U.flatgrad(optimgain, var_list)])
    compute_fvp = U.function([flat_tangent, ob, ac, atarg], fvp)
    compute_vflossandgrad = U.function([ob, ret], U.flatgrad(vferr, vf_var_list))

    U.initialize()
    th_init = get_flat()
    MPI.COMM_WORLD.Bcast(th_init, root=0)
    set_from_flat(th_init)
    d_adam.sync()
    vfadam.sync()

    # Prepare for rollouts
    # ----------------------------------------

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()

    writer = [tf.summary.FileWriter(log_dir + "/trpo/eval_" + str(i)) for i in range(style_num+1)]

    metrics_name = ["True_rewards", "Run_step", "Decision_step", "Change_step",\
                      "Max_acc", "Min_acc", "Mp_acc", "Mn_acc", "Vp_acc", "Vn_acc",\
                      "Max_spd", "Min_spd", "M_spd", "V_spd", "Num_pos_time", "Num_neg_time",\
                      "Gap_dis_lead", "Gap_dis_tail", "Gap_dis_min", "Gap_ind", "Done_ratio"]

    prev_eval_metrics = [ np.array([0]*len(metrics_name)) for _ in range(style_num+1) ]

    loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]
    reward_names = ["dis_reward", "tot_reward"]
    g_loss_stats = stats(loss_names)
    d_loss_stats = stats(discriminator.loss_name)
    reward_stats = stats(reward_names)
    ep_stats = stats(metrics_name)

    # if provide pretrained weight
    if load_path is not None:
        # mlearner.load_model(load_path)
        U.load_state(load_path)

    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0]) == 1
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

        # pi_before_paras, dis_before_paras = save_para(pi_para_model, dis_para_model)
        # pi_after_paras = []
        # dis_after_paras = []
        # meta_stepsize *= (1 - iters_so_far/max_iters)

        for style_index in range(style_num):

            for inner_iter in range(update_times):

                logger.log("********** Style %i  Inner Iterations %i Outer Iterations %i ************" % (
                style_index, inner_iter, iters_so_far))

                # ------------------ Sampling------------------
                with timed("sampling", rank):
                    segs, evals = traj_segment_generator(pi, env, path_num, stochastic=True,
                                                         style_index=style_index)  # Iterater
                    evals["metrics"] = get_eval_metrics(evals["metrics"])

                for seg in segs:
                    seg["lprobs"] = compute_path_probs(seg["ac"], seg["mean"], seg["logstd"], ac_space, is_expert=False)

                ob = np.concatenate([seg["ob"] for seg in segs])
                ac = np.concatenate([seg["ac"] for seg in segs])
                mean = np.concatenate([seg["mean"] for seg in segs])
                std = np.concatenate([seg["logstd"] for seg in segs])
                lprobs = np.concatenate([seg["lprobs"] for seg in segs])

                # ------------------ Update D ------------------
                logger.log("Optimizing Discriminator...")
                logger.log(fmt_row(13, discriminator.loss_name))
                batch_size = len(ob) // d_iters

                labels = np.zeros((batch_size * 2, 1))
                labels[batch_size:] = 1.0

                d_losses = []
                for ob_batch, ac_batch, lprobs_batch in dataset.iterbatches((ob, ac, lprobs), shuffle=True,
                                                                            include_final_partial_batch=False,
                                                                            batch_size=batch_size):
                    ob_expert, ac_expert, mean_expert, std_expert = expert_dataset[style_index].get_next_batch(len(ob_batch))
                    lprobs_expert = compute_path_probs(ac_expert, mean_expert, std_expert, ac_space, is_expert=True)
                    # update running mean/std for discriminator
                    if hasattr(discriminator, "obs_rms"): discriminator.obs_rms.update(
                        np.concatenate((ob_batch, ob_expert), 0))
                    lprob_batch = np.concatenate([lprobs_batch, lprobs_expert], axis=0)

                    *newlosses, g = discriminator.lossandgrad(ob_batch, ac_batch, ob_expert, ac_expert, labels,
                                                                   lprob_batch)
                    d_adam.update(allmean(g, nworkers), d_stepsize)
                    d_losses.append(newlosses)

                logger.log(fmt_row(13, np.mean(d_losses, axis=0)))

                # ------------------ get Reward and Advantage ------------------

                Disc_rewards = []
                Total_rewards = []

                for seg in segs:
                    new_batch_size = seg["ob"].shape[0]
                    lprobs_rew = seg["lprobs"]
                    Disc_reward = discriminator.get_reward(seg["ob"], seg["ac"], new_batch_size, lprobs_rew)

                    rewards = coe_d * Disc_reward

                    if use_true_reward:
                        true_rew = np.expand_dims(seg["true_reward"], axis=1)
                        rewards += coe_t * true_rew  ##reward_2

                    elif use_sparse_reward:
                        print('rewards: ', rewards)
                        sparse_rew = np.expand_dims(seg["sparse_reward"], axis=1)
                        print("sparse: ", sparse_rew)
                        rewards += coe_t * sparse_rew  ##reward_2

                    seg["rew"] = rewards

                    Disc_rewards.append(np.sum(Disc_reward))
                    Total_rewards.append(np.sum(rewards))
                    add_vtarg_and_adv(seg, gamma, lam)

                Disc_reward = np.array(Disc_rewards).mean()
                Total_reward = np.array(Total_rewards).mean()

                # ------------------ Update G ------------------
                def fisher_vector_product(p):
                    return allmean(compute_fvp(p, *fvpargs), nworkers) + cg_damping * p

                logger.log("Optimizing Policy...")
                atarg = np.concatenate([seg["adv"] for seg in segs])
                tdlamret = np.concatenate([seg["tdlamret"] for seg in segs])
                vpredbefore = np.concatenate([seg["vpred"] for seg in segs])

                atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate

                if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob)  # update running mean/std for policy

                args = ob, ac, atarg
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

                # ------------------ Update V ------------------
                with timed("vf", rank):
                    for _ in range(vf_iters):
                        for (mbob, mbret) in dataset.iterbatches((ob, tdlamret), shuffle=True,
                                                                        include_final_partial_batch=False,
                                                                        batch_size=128):
                            if hasattr(pi, "ob_rms"):
                                pi.ob_rms.update(mbob)  # update running mean/std for policy
                            g = allmean(compute_vflossandgrad(mbob, mbret), nworkers)
                            vfadam.update(g, vf_stepsize)

                g_losses = meanlosses
                for (lossname, lossval) in zip(loss_names, meanlosses):
                    logger.record_tabular(lossname, lossval)
                logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))



            prev_eval_metrics[style_index] = summary_write(rank, writer[style_index], g_loss_stats, g_losses, d_loss_stats,
                       d_losses, reward_stats, Disc_reward, Total_reward, evals, ep_stats, iters_so_far, prev_eval_metrics[style_index])

        #     pi_after_para, dis_after_para = save_para(pi_para_model, dis_para_model)
        #     pi_after_paras.append(pi_after_para)
        #     dis_after_paras.append(dis_after_para)
        #     restore_para(pi_para_model, dis_para_model, pi_before_paras, dis_before_paras)
        #
        # meta_update(pi_para_model, dis_para_model, meta_stepsize, pi_after_paras, dis_after_paras, pi_before_paras, dis_before_paras)

        # ------------------ Online Test ------------------

        # if iters_so_far%20==0:
        #
        #     pi_before_paras, dis_before_paras = save_para(pi_para_model, dis_para_model)
        #
        #     style_index = style_num
        #     for inner_iter in range(eval_update_times):
        #         d_losses, g_losses, D_reward, T_reward, _ = inner_loop(env, path_num, rank, style_index, inner_iter, iters_so_far, ac_space, discriminator, d_iters, mlearner, expert_dataset, d_stepsize, coe_d, coe_t, use_true_reward, use_sparse_reward,
        #                                     gamma, lam, cg_damping, max_kl, cg_iters, loss_names, vf_iters, vf_stepsize)
        #
        #     evals = runner(env, path_num, pi=pi, style_num=style_num)
        #
        #     prev_eval_metrics[style_index] = summary_write(rank, writer[style_index], g_loss_stats, g_losses,
        #                                                    d_loss_stats, d_losses, reward_stats, D_reward, T_reward,
        #                                                    evals, ep_stats, iters_so_far, prev_eval_metrics[style_index])
        #
        #     restore_para(pi_para_model, dis_para_model, pi_before_paras, dis_before_paras)

        # ------------------ Log info ------------------

        logger.record_tabular("EpThisIter", path_num)
        iters_so_far += 1
        if iters_so_far > 45300:
            break
        logger.record_tabular("TimeElapsed", time.time() - tstart)

        if rank == 0:
            logger.dump_tabular()




def summary_write(rank, writer, g_loss_stats, g_losses, d_loss_stats, d_losses, reward_stats, D_reward, T_reward, evals, ep_stats, iters_so_far, prev_eval_metrics):

    if rank == 0:

        g_loss_stats.add_all_summary(writer, g_losses, iters_so_far)
        d_loss_stats.add_all_summary(writer, np.mean(d_losses, axis=0), iters_so_far)
        reward_stats.add_all_summary(writer, np.array([D_reward, T_reward]), iters_so_far)

    temp = evals
    metric = temp["metrics"]
    step_num = extract_num(metric["gaps"]["gap_idx_hist"])

    lrlocal = [temp["ep_rets"], metric["steps"]["run_step"], metric["steps"]["decision_step"], metric["steps"]["change_step"], \
               metric["acces"]["max"], metric["acces"]["min"], metric["acces"]["mean_pos"], metric["acces"]["mean_neg"], metric["acces"]["var_pos"], metric["acces"]["var_neg"], \
               metric["speeds"]["max"], metric["speeds"]["min"], metric["speeds"]["mean"], metric["speeds"]["var"], metric["times"]["num_pos_acce"], metric["times"]["num_neg_acce"], \
               metric["gaps"]["dis_lead"], np.array(metric["gaps"]["dis_tail"]), np.array(metric["gaps"]["dis_min"])]  # local values
    lrlocal.append(step_num)
    lrlocal.append([temp["done_ratio"]])

    # sliding with last evaluation metrics
    eval_metrics = np.array([np.mean(elem) for elem in lrlocal])
    eval_show = list(0.5 * prev_eval_metrics + 0.5 * eval_metrics)
    prev_eval_metrics = eval_metrics

    # print([np.mean(elem) for elem in lrlocal])
    if rank == 0:
        ep_stats.add_all_summary(writer, eval_show, iters_so_far)

    return prev_eval_metrics

def save_para(pi_para_model, dis_para_model):
    pi_paras = pi_para_model.export_variables()
    dis_paras = dis_para_model.export_variables()
    return pi_paras, dis_paras

def restore_para(pi_para_model, dis_para_model, pi_before_paras, dis_before_paras):
    pi_para_model.import_variables(pi_before_paras)
    dis_para_model.import_variables(dis_before_paras)

def meta_update(pi_para_model, dis_para_model, meta_stepsize, pi_after_paras, dis_after_paras, pi_before_paras, dis_before_paras):
    pi_after_paras = average_vars(pi_after_paras)
    dis_after_paras = average_vars(dis_after_paras)
    pi_para_model.import_variables(interpolate_vars(pi_before_paras, pi_after_paras, meta_stepsize))
    dis_para_model.import_variables(interpolate_vars(dis_before_paras, dis_after_paras, meta_stepsize))