import os
from mpi4py import MPI
import tensorflow as tf
from baselines.common import explained_variance, fmt_row
from baselines import logger
from baselines.gail.statistics import stats
from train_util import *
from MetaLearner import MetaLearner
from run import runner
from baselines.common import tf_util as U


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

    print("pi_id: ", id(pi))
    print("oldpi_id: ", id(oldpi))
    print("discrimitor: ", id(discriminator))

    mlearner = MetaLearner(env, nworkers, rank, pi, oldpi, discriminator, entcoeff)

    mlearner.initialize()

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
        mlearner.load_model(load_path)

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

        pi_before_paras, dis_before_paras = mlearner.save_para()
        # print("1: ", pi_before_paras[0])
        # print("2: ", dis_before_paras[0])
        pi_after_paras = []
        dis_after_paras = []
        cur_meta_stepsize = meta_stepsize * (1 - iters_so_far/max_iters)

        logger_done, logger_ret = [], []

        for style_index in range(style_num):

            # p, d = mlearner.save_para()
            # print("3: ", p[0])
            # print("4: ", d[0])

            for inner_iter in range(update_times):

                # print("dis_before: ", U.get_session().run(mlearner.dis_var_list[0]))

                d_losses, g_losses, D_reward, T_reward, evals = inner_loop(env, path_num, rank, style_index, inner_iter, iters_so_far, ac_space, discriminator, d_iters, mlearner, expert_dataset, d_stepsize, coe_d, coe_t, use_true_reward, use_sparse_reward,
                                            gamma, lam, cg_damping, max_kl, cg_iters, loss_names, vf_iters, vf_stepsize)

                # print("dis_after: ", U.get_session().run(mlearner.dis_var_list[0]))

            prev_eval_metrics[style_index], l_r, l_d = summary_write(rank, writer[style_index], g_loss_stats, g_losses, d_loss_stats,
                       d_losses, reward_stats, D_reward, T_reward, evals, ep_stats, iters_so_far, prev_eval_metrics[style_index])

            logger_done.append(l_d)
            logger_ret.append(round(l_r, 2))

            pi_after_para, dis_after_para = mlearner.save_para()
            pi_after_paras.append(pi_after_para)
            dis_after_paras.append(dis_after_para)
            # print("5: ", pi_after_para[0])
            # print("6: ", dis_after_para[0])
            # print("7: ", len(pi_after_paras))
            # print("8: ", len(dis_after_paras))
            mlearner.restore_para(pi_before_paras, dis_before_paras)
        #
        mlearner.meta_update(cur_meta_stepsize, pi_after_paras, dis_after_paras, pi_before_paras, dis_before_paras)

        # ------------------ Online Test ------------------

        if iters_so_far%20==0:

            pi_before_paras, dis_before_paras = mlearner.save_para()

            style_index = style_num
            for inner_iter in range(eval_update_times):
                d_losses, g_losses, D_reward, T_reward, _ = inner_loop(env, path_num, rank, style_index, inner_iter, iters_so_far, ac_space, discriminator, d_iters, mlearner, expert_dataset, d_stepsize, coe_d, coe_t, use_true_reward, use_sparse_reward,
                                            gamma, lam, cg_damping, max_kl, cg_iters, loss_names, vf_iters, vf_stepsize)

            evals = runner(env, path_num*2, pi=mlearner.pi, style_num=style_num)

            prev_eval_metrics[style_index], l_r, l_d = summary_write(rank, writer[style_index], g_loss_stats, g_losses,
                                                           d_loss_stats, d_losses, reward_stats, D_reward, T_reward,
                                                           evals, ep_stats, iters_so_far, prev_eval_metrics[style_index])

            logger_done.append(l_d)
            logger_ret.append(round(l_r, 2))

            mlearner.restore_para(pi_before_paras, dis_before_paras)

        # ------------------ Log info ------------------

        logger.record_tabular("EpThisIter", path_num)
        iters_so_far += 1
        if iters_so_far > 45300:
            break
        logger.record_tabular("TimeElapsed", time.time() - tstart)

        logger.record_tabular("ep_rets", logger_ret)
        logger.record_tabular("done_ratio", logger_done)

        if rank == 0:
            logger.dump_tabular()



def inner_loop(env, path_num, rank, style_index, inner_iter, iters_so_far, ac_space, discriminator, d_iters, mlearner, expert_dataset, d_stepsize, coe_d, coe_t, use_true_reward, use_sparse_reward,
               gamma, lam, cg_damping, max_kl, cg_iters, loss_names, vf_iters, vf_stepsize):

    logger.log("********** Style %i  Inner Iterations %i Outer Iterations %i ************" % (style_index, inner_iter, iters_so_far))

    # ------------------ Sampling------------------
    with timed("sampling", rank):
        segs, evals = traj_segment_generator(mlearner.pi, env, path_num, stochastic=True, style_index=style_index)  # Iterater
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

    d_losses = mlearner.update_D(ob, ac, lprobs, batch_size, expert_dataset[style_index], labels, d_stepsize)
    logger.log(fmt_row(13, np.mean(d_losses, axis=0)))

    # ------------------ get Reward and Advantage ------------------

    Disc_reward, Total_reward = mlearner.getRewardAndAdvantage(segs, coe_d, coe_t, use_true_reward, use_sparse_reward,
                                                                 gamma, lam)


    # ------------------ Update G ------------------
    logger.log("Optimizing Policy...")
    atarg = np.concatenate([seg["adv"] for seg in segs])
    tdlamret = np.concatenate([seg["tdlamret"] for seg in segs])
    vpredbefore = np.concatenate([seg["vpred"] for seg in segs])

    atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate

    g_losses = mlearner.update_P(ob, ac, atarg, cg_damping, max_kl, cg_iters)


    for (lossname, lossval) in zip(loss_names, g_losses):
        logger.record_tabular(lossname, lossval)

    # ------------------ Update V ------------------
    mlearner.update_V(vf_iters, ob, tdlamret, vf_stepsize)
    logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))

    return d_losses, g_losses, Disc_reward, Total_reward, evals



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

    return prev_eval_metrics, np.mean(temp["ep_rets"]), temp["done_ratio"]