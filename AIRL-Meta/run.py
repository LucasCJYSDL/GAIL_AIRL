
'''
Disclaimer: this code is highly based on trpo_mpi at @openai/baselines and @openai/imitation
'''

import argparse
import os
import logging
from mpi4py import MPI
from tqdm import tqdm

import numpy as np
import gym

import mlp_policy
from baselines.common import set_global_seeds, tf_util as U
from baselines.common.misc_util import boolean_flag
from baselines import bench
from baselines import logger

from dataset.dset import AutoDrive_Dset
from adversary import TransitionClassifier
import matplotlib.pyplot as plt
import gymAutoDrive
import tensorflow as tf
from baselines.gail.statistics import stats
from train_util import get_eval_metrics, extract_num

def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of GAIL")
    parser.add_argument('--env_id', help='environment ID', default='AutoDrive-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--expert_path', type=str, default='./data/')
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='checkpoint')
    parser.add_argument('--log_dir', help='the directory to save log file', default='log')
    parser.add_argument('--load_model_ind', help='if positive, load the corresponding model', type=int, default=-1)
    # Task
    parser.add_argument('--task', type=str, choices=['train', 'evaluate', 'sample'], default='train')
    parser.add_argument('--exp_ind', help='index of the experiment', type=int, default=57)
    # for evaluatation
    boolean_flag(parser, 'stochastic_policy', default=False, help='use stochastic/deterministic policy to evaluate')
    boolean_flag(parser, 'save_sample', default=True, help='save the trajectories or not')
    parser.add_argument('--num_of_trajs', type=int, default=3000)
    #  Mujoco Dataset Configuration
    parser.add_argument('--traj_limitation', type=int, default=-1)
    # Optimization Configuration
    parser.add_argument('--d_iters', help='number of iterations to train discriminator in each epoch', type=int, default=1)
    parser.add_argument('--vf_iters', help='number of iterations to train value net in each epoch', type=int, default=5)
    # Network Configuration (Using MLP Policy)
    parser.add_argument('--policy_hidden_size', type=int, default=100)
    parser.add_argument('--adversary_hidden_size', type=int, default=256)
    # Algorithms Configuration
    parser.add_argument('--algo', type=str, choices=['trpo', 'ppo'], default='trpo')
    parser.add_argument('--max_kl', type=float, default=0.01)
    parser.add_argument('--policy_entcoeff', help='entropy coefficiency of policy', type=float, default=0.01)
    parser.add_argument('--adversary_entcoeff', help='entropy coefficiency of discriminator', type=float, default=1e-3)
    parser.add_argument('--use_default', help='whether to use the reward function defined in the paper(AIRL)',
                        default=True)
    # Reward Configuration
    parser.add_argument('--coe_d', help='coefficient for the disriminator', type=float, default=1.0)
    parser.add_argument('--coe_t', help='coefficient for the true reward', type=int, default=1.0)
    # Traing Configuration
    parser.add_argument('--save_per_iter', help='save model every xx iterations', type=int, default=20)
    parser.add_argument('--num_iters', help='number of timesteps per episode', type=int, default=45000)
    parser.add_argument('--style_num', help='number of styles we want to learn', type=int, default=2)
    parser.add_argument('--meta_step', help='step_size for meta-learning', type=float, default=0.25)
    parser.add_argument('--path_num', help='number of tragetories we want to collect', type=int, default=10)
    parser.add_argument('--update_times', help='number of update iterations for inner loop', type=int, default=2)
    parser.add_argument('--eval_update_times', help='number of update iterations for inner loop when testing', type=int, default=4)
    parser.add_argument('--use_true_reward',
                        help='whether to add the true reward(get from env.step) to the reward function', default=True)

    parser.add_argument('--use_sparse_reward', help='whether to add the sparse reward(get from env.step) to the reward function', default=False)
    parser.add_argument('--random', help='whether to randomize the expert data', default=True)

    return parser.parse_args()


def get_task_name(args):

    task_name = args.env_id.split("-")[0] + '_' + str(args.exp_ind)
    return task_name

def get_load_path(args):
    if args.load_model_ind == -1:
        return None
    load_path = args.checkpoint_dir + '/' +get_task_name(args) + str(args.load_model_ind)

    return load_path


def main(args):
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(args.seed)
    env = gym.make(args.env_id)

    anim_episode = 3
    env.init_global_parameters_for_gather_data(anim_episode)

    def policy_fn(name, ob_space, ac_space, reuse=False):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    reuse=reuse, hid_size=args.policy_hidden_size, num_hid_layers=2)

    #env = bench.Monitor(env, logger.get_dir() and
    #                    osp.join(logger.get_dir(), "monitor.json"), allow_early_resets=False)

    env.seed(args.seed)

    gym.logger.setLevel(logging.WARN)
    task_name = get_task_name(args)
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, task_name)
    args.log_dir = os.path.join(args.log_dir, task_name)

    if args.task == 'train':
        dataset = []
        for i in range(args.style_num+1):
            expert_path = args.expert_path + env.spec.id + '_' + str(i) + ".npz"
            dataset.append(AutoDrive_Dset(expert_path=expert_path, traj_limitation=args.traj_limitation, randomize=args.random))
        discriminator = TransitionClassifier(env, args.adversary_hidden_size, use_default=args.use_default,
                                            entcoeff=args.adversary_entcoeff)
        train(env,
              args.seed,
              policy_fn,
              discriminator,
              dataset,
              args.algo,
              args.d_iters,
              args.vf_iters,
              args.policy_entcoeff,
              args.num_iters,
              args.save_per_iter,
              args.checkpoint_dir,
              args.log_dir,
              get_load_path(args),
              args.use_true_reward, args.use_sparse_reward,
              args.coe_d, args.coe_t,
              args.style_num,
              args.path_num,
              args.update_times,
              args.eval_update_times,
              args.meta_step,
              task_name
              )

    elif args.task == 'evaluate':

        runner(env,
               policy_fn,
               number_trajs=args.num_of_trajs,
               save=args.save_sample,
               style_num=args.style_num
               )

    elif args.task == 'sample':

        sampler(env,
                number_trajs=args.num_of_trajs,
                save=args.save_sample,
                style_num=args.style_num
                )

    else:
        raise NotImplementedError
    env.close()

def train(env, seed, policy_fn, discriminator, dataset, algo, d_iters, vf_iters, policy_entcoeff, num_iters, save_per_iter,
          checkpoint_dir, log_dir, load_path, use_true_reward, use_sparse_reward, coe_d, coe_t, style_num, path_num, update_times, eval_update_times, meta_stepsize, task_name=None):


    if algo == 'trpo':
        import trpo_mpi, test_trpo_mpi
        # Set up for MPI seed
        rank = MPI.COMM_WORLD.Get_rank()
        if rank != 0:
            logger.set_level(logger.DISABLED)
        workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
        set_global_seeds(workerseed)
        env.seed(workerseed)
        trpo_mpi.learn(env, policy_fn, discriminator, dataset, rank,
                       d_iters=d_iters,
                       entcoeff=policy_entcoeff,
                       max_iters=num_iters,
                       ckpt_dir=checkpoint_dir, log_dir=log_dir,
                       save_per_iter=save_per_iter,
                       max_kl=0.01, cg_iters=10, cg_damping=0.1, gamma=0.995, lam=0.97,
                       coe_d=coe_d, coe_t=coe_t, style_num=style_num, path_num=path_num, update_times=update_times, eval_update_times=eval_update_times,
                       meta_stepsize=meta_stepsize, vf_iters=vf_iters, vf_stepsize=1e-3, load_path=load_path, use_true_reward=use_true_reward,
                       use_sparse_reward=use_sparse_reward,
                       task_name=task_name)

        # test_trpo_mpi.learn(env, policy_fn, discriminator, dataset, rank,
        #                    d_iters=d_iters,
        #                    entcoeff=policy_entcoeff,
        #                    max_iters=num_iters,
        #                    ckpt_dir=checkpoint_dir, log_dir=log_dir,
        #                    save_per_iter=save_per_iter,
        #                    max_kl=0.01, cg_iters=10, cg_damping=0.1, gamma=0.995, lam=0.97,
        #                    coe_d=coe_d, coe_t=coe_t, style_num=style_num, path_num=path_num, update_times=update_times, eval_update_times=eval_update_times,
        #                    meta_stepsize=meta_stepsize, vf_iters=vf_iters, vf_stepsize=1e-3, load_path=load_path, use_true_reward=use_true_reward,
        #                    use_sparse_reward=use_sparse_reward,
        #                    task_name=task_name)
    else:
        raise NotImplementedError


#Todo
def runner(env, number_trajs, policy_func=None, pi=None, save=False, style_num=2, reuse=False):
    # Setup network
    # ----------------------------------------

    if pi:
        style_index = style_num
        obs, acs, evals = traj_segment_sampler(env, number_trajs, style_index, pi)
        evals["metrics"] = get_eval_metrics(evals["metrics"])
        return evals

    else:
        ob_space = env.observation_space
        ac_space = env.action_space

        pi = policy_func("pi", ob_space, ac_space, reuse=reuse)

        logdir = args.log_dir+"/log.txt"
        f = open(logdir, 'a')

        inxs = [100, 200] #input the indexs you want to test

        for inx in inxs:

            load_model_path = args.checkpoint_dir + '/' +get_task_name(args) + str(inx)

            print(load_model_path)

            U.initialize()
        # Prepare for rollouts
        # ----------------------------------------
            U.load_state(load_model_path)

            sampler(env, number_trajs, save, style_num, pi, inx)


def sampler(env, number_trajs, save=False, style_num=2, pi=None, inx=-1):

    obs_list, acs_list, evals = [], [], []
    for i in range(style_num+1):
        obs, acs, eval = traj_segment_sampler(env, number_trajs, style_index=i, pi=pi)
        obs_list.append(obs)
        acs_list.append(acs)
        evals.append(eval)

    for i in range(len(evals)):
        evals[i]["metrics"] = get_eval_metrics(evals[i]["metrics"])

    lrlocals = []
    lrlocal_array = []
    for i in range(style_num+1):
        temp = evals[i]
        metric = temp["metrics"]
        step_num = extract_num(metric["gaps"]["gap_idx_hist"])

        lrlocal = [temp["ep_rets"], metric["steps"]["run_step"], metric["steps"]["decision_step"], metric["steps"]["change_step"], \
                   metric["acces"]["max"], metric["acces"]["min"], metric["acces"]["mean_pos"], metric["acces"]["mean_neg"], metric["acces"]["var_pos"], metric["acces"]["var_neg"], \
                   metric["speeds"]["max"], metric["speeds"]["min"], metric["speeds"]["mean"], metric["speeds"]["var"], metric["times"]["num_pos_acce"], metric["times"]["num_neg_acce"], \
                   metric["gaps"]["dis_lead"], np.array(metric["gaps"]["dis_tail"]), np.array(metric["gaps"]["dis_min"])]  # local values

        lrlocal.append(step_num)
        lrlocal.append([temp["done_ratio"]])
        lrlocal_array.append(lrlocal)
        lrlocal = list(map(np.mean, lrlocal))
        lrlocals.append(lrlocal)

    ep_stats = ["True_rewards", "Run_step", "Decision_step", "Change_step", \
                "Max_acc", "Min_acc", "Mp_acc", "Mn_acc", "Vp_acc", "Vn_acc", \
                "Max_spd", "Min_spd", "M_spd", "V_spd", "Num_pos_time", "Num_neg_time", \
                "Gap_dis_lead", "Gap_dis_tail", "Gap_dis_min", "Gap_ind", "Done_ratio"]
    lenth = len(ep_stats)

    #save evaluation data
    #print("lrlocal_array: ", lrlocal_array)
    for i in range(style_num+1):
        fname = "Style_"+str(i)
        if inx!=-1:
            fdir = "evaluate"+"/"+str(inx)+"/"
            if not os.path.exists(fdir):
                os.makedirs(fdir)
        else:
            fdir = "./collection/"
            if not os.path.exists(fdir):
                os.makedirs(fdir)

        fname=fdir+fname
        np.savez(fname,True_rewards=lrlocal_array[i][0], Run_step=lrlocal_array[i][1], Decision_step=lrlocal_array[i][2],
                 Change_step=lrlocal_array[i][3], Max_acc=lrlocal_array[i][4], Min_acc=lrlocal_array[i][5], Mp_acc=lrlocal_array[i][6],
                 Mn_acc=lrlocal_array[i][7], Vp_acc=lrlocal_array[i][8], Vn_acc=lrlocal_array[i][9], Max_spd=lrlocal_array[i][10],
                 Min_spd=lrlocal_array[i][11], M_spd=lrlocal_array[i][12], V_spd=lrlocal_array[i][13], Num_pos_time=lrlocal_array[i][14],
                 Num_neg_time=lrlocal_array[i][15], Gap_dis_lead=lrlocal_array[i][16], Gap_dis_tail=lrlocal_array[i][17],
                 Gap_dis_min=lrlocal_array[i][18], Gap_ind=lrlocal_array[i][19], Done_ratio=lrlocal_array[i][20])

    #plot data
    #print("lrlocals: ", lrlocals)

    x_data = ("style_0", "style_1", "style_2")
    colors = ["SkyBlue", "IndianRed", "yellow"]

    for j in range(lenth):
        fig, ax = plt.subplots()
        for i in range(style_num+1):
            ax.bar(i*0.5, [lrlocals[i][j]], 0.5, color=colors[i],label=x_data[i])
        ax.set_title(ep_stats[j])
        plt.xticks(np.arange(style_num+1)*0.5, x_data)
        pic_dir = "./images/"
        if not os.path.exists(pic_dir):
            os.makedirs(pic_dir)
        plt.savefig(pic_dir+ep_stats[j]+".png")
        plt.close()

    #save expert data
    if save and not pi:
        for i in range(style_num+1):
            idx = np.arange(len(obs_list[i]))
            np.random.shuffle(idx)
            obs_list_i = np.array(obs_list[i])[idx]
            acs_list_i = np.array(acs_list[i])[idx]
            d_dir = "./data/"
            if not os.path.exists(d_dir):
                os.makedirs(d_dir)
            filename = d_dir + env.spec.id + '_' + str(i)
            np.savez(filename, obs=np.array(obs_list_i), acs=np.array(acs_list_i))


def traj_segment_sampler(env, path_num, style_index, pi=None, stochastic=False):

    done_num = 0
    ep_true_rets, metrics = [],[]
    obs_list, acs_list = [], []

    for i in tqdm(range(path_num)):

        env.set_ego_driving_style(style_index)

        ac = env.action_space.sample()
        ob_vars = env.reset()
        ob = env.ego_vehicle.get_ob_features(env.vehicle_list_separ_lanes)
        # Initialize history arrays
        obs = []
        true_rews = []
        acs = []

        while True:
            if pi:
                ac, _, _, _ = pi.act(stochastic, ob)
                ac = [ac]
            else:
                ac = env.temp_agent_act(ob)
            obs.append(ob)
            acs.append(ac)

            ob_vars, true_rew, ego_done, agent_info = env.step(ac)
            ob = env.ego_vehicle.get_ob_features(ob_vars)
            true_rews.append(true_rew)
            new = agent_info['break']

            if new:
                metrics.append(agent_info['episode']['metric'])
                if agent_info['episode']['l'] and agent_info['episode']['r']:
                    done_num += 1
                    print(agent_info['episode'])
                    ep_true_rets.append(agent_info['episode']['r'])
                obs_list.append(np.array(obs))
                acs_list.append(np.array(acs))
                break


    evals = {"done_ratio": done_num/path_num, "ep_rets": ep_true_rets, "metrics": metrics}

    return obs_list, acs_list, evals


if __name__ == '__main__':
    args = argsparser()
    main(args)
