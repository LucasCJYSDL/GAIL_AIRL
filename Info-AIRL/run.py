
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
from posterior import Posterior
from train_util import get_eval_metrics, extract_num

def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of GAIL")
    parser.add_argument('--env_id', help='environment ID', default='AutoDrive-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--expert_path', type=str, default='./data/AutoDrive-v0-6000.npz')
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='checkpoint')
    parser.add_argument('--log_dir', help='the directory to save log file', default='log')
    parser.add_argument('--load_model_ind', help='if positive, load the corresponding model', type=int, default=-1)
    # Task
    parser.add_argument('--task', type=str, choices=['train', 'evaluate', 'sample'], default='evaluate')
    parser.add_argument('--exp_ind', help='index of the experiment', type=int, default=57)
    # for evaluatation
    boolean_flag(parser, 'stochastic_policy', default=False, help='use stochastic/deterministic policy to evaluate')
    boolean_flag(parser, 'save_sample', default=False, help='save the trajectories or not')
    parser.add_argument('--num_of_trajs', type=int, default=10)
    #  Mujoco Dataset Configuration
    parser.add_argument('--traj_limitation', type=int, default=-1)
    # Optimization Configuration
    parser.add_argument('--d_step', help='number of steps to train discriminator in each epoch', type=int, default=2)
    parser.add_argument('--d_iters', help='number of iterations to train discriminator in each epoch', type=int, default=1)
    parser.add_argument('--p_iters', help='number of iterations to train posterior in each epoch', type=int, default=3)
    parser.add_argument('--vf_iters', help='number of iterations to train value net in each epoch', type=int, default=5)
    # Network Configuration (Using MLP Policy)
    parser.add_argument('--policy_hidden_size', type=int, default=100)
    parser.add_argument('--adversary_hidden_size', type=int, default=256)
    parser.add_argument('--posterior_hidden_size', type=int, default=256)
    # Algorithms Configuration
    parser.add_argument('--algo', type=str, choices=['trpo', 'ppo'], default='trpo')
    parser.add_argument('--max_kl', type=float, default=0.01)
    parser.add_argument('--policy_entcoeff', help='entropy coefficiency of policy', type=float, default=0.01)
    parser.add_argument('--adversary_entcoeff', help='entropy coefficiency of discriminator', type=float, default=1e-3)
    parser.add_argument('--use_default', help='whether to use the reward function defined in the paper(AIRL)',
                        default=True)
    # Reward Configuration
    parser.add_argument('--coe_d', help='coefficient for the disriminator', type=float, default=1.0)
    parser.add_argument('--coe_p', help='coefficient for the posterior', type=int, default=1.5)
    parser.add_argument('--coe_t', help='coefficient for the true reward', type=int, default=1.0)
    # Traing Configuration
    parser.add_argument('--save_per_iter', help='save model every xx iterations', type=int, default=100)
    parser.add_argument('--num_iters', help='number of timesteps per episode', type=int, default=45000)
    parser.add_argument('--buffer_size', help='size of the training buffer', type=int, default=40)
    parser.add_argument('--sample_size', help='number of trajectories for training', type=int, default=20)
    parser.add_argument('--encode_num', help='number of styles we want to learn', type=int, default=3)
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

    def policy_fn(name, ob_space, ac_space, encode_num, reuse=False):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    reuse=reuse, hid_size=args.policy_hidden_size, num_hid_layers=2, encode_num=encode_num)

    #env = bench.Monitor(env, logger.get_dir() and
    #                    osp.join(logger.get_dir(), "monitor.json"), allow_early_resets=False)

    env.seed(args.seed)

    gym.logger.setLevel(logging.WARN)
    task_name = get_task_name(args)
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, task_name)
    args.log_dir = os.path.join(args.log_dir, task_name)

    if args.task == 'train':
        dataset = AutoDrive_Dset(expert_path=args.expert_path, traj_limitation=args.traj_limitation, randomize=args.random)
        discriminator = TransitionClassifier(env, args.adversary_hidden_size, use_default=args.use_default,
                                            entcoeff=args.adversary_entcoeff)
        posterior = Posterior(env, args.posterior_hidden_size, args.encode_num)
        posterior_target = Posterior(env, args.posterior_hidden_size, args.encode_num, scope="target")
        train(env,
              args.seed,
              policy_fn,
              discriminator,
              posterior,
              posterior_target,
              dataset,
              args.algo,
              args.d_step,
              args.d_iters,
              args.p_iters,
              args.vf_iters,
              args.policy_entcoeff,
              args.num_iters,
              args.save_per_iter,
              args.checkpoint_dir,
              args.log_dir,
              get_load_path(args),
              args.use_true_reward, args.use_sparse_reward,
              args.buffer_size, args.sample_size,
              args.coe_d, args.coe_p, args.coe_t,
              args.encode_num,
              task_name
              )
    elif args.task == 'evaluate':

        runner(env,
               policy_fn,
               number_trajs=args.num_of_trajs,
               save=args.save_sample,
               encode_num=args.encode_num
               )

    elif args.task == 'sample':

        sampler(env,
                number_trajs=args.num_of_trajs,
                save=args.save_sample,
                encode_num=args.encode_num
                )

    else:
        raise NotImplementedError
    env.close()

def train(env, seed, policy_fn, discriminator, posterior, posterior_target, dataset, algo,
          d_step, d_iters, p_iters, vf_iters, policy_entcoeff, num_iters, save_per_iter,
          checkpoint_dir, log_dir, load_path, use_true_reward, use_sparse_reward,buffer_size, sample_size, coe_d, coe_p, coe_t, encode_num, task_name=None):


    if algo == 'trpo':
        import trpo_mpi
        # Set up for MPI seed
        rank = MPI.COMM_WORLD.Get_rank()
        if rank != 0:
            logger.set_level(logger.DISABLED)
        workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
        set_global_seeds(workerseed)
        env.seed(workerseed)
        trpo_mpi.learn(env, policy_fn, discriminator, dataset, rank,
                       posterior=posterior, posterior_target=posterior_target,
                       d_step=d_step, d_iters=d_iters, p_iters=p_iters,
                       entcoeff=policy_entcoeff,
                       max_iters=num_iters,
                       ckpt_dir=checkpoint_dir, log_dir=log_dir,
                       save_per_iter=save_per_iter,
                       max_kl=0.01, cg_iters=10, cg_damping=0.1, gamma=0.995, lam=0.97,
                       buffer_size=buffer_size, sample_size=sample_size, coe_d=coe_d, coe_p=coe_p, coe_t=coe_t, encode_num=encode_num,
                       vf_iters=vf_iters, vf_stepsize=1e-3, load_path=load_path, use_true_reward=use_true_reward,
                       use_sparse_reward=use_sparse_reward,
                       task_name=task_name)
    else:
        raise NotImplementedError


#Todo
def runner(env, policy_func, number_trajs, save=False, encode_num=2, reuse=False):
    # Setup network
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space

    pi = policy_func("pi", ob_space, ac_space, encode_num, reuse=reuse)

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

        sampler(env, number_trajs, save, encode_num, pi, inx)


def sampler(env, number_trajs, save=False, encode_num=2, pi=None, inx=-1):

    obs_list, acs_list, evals = traj_segment_sampler(env, encode_num * number_trajs, encode_num=encode_num, pi=pi)
    for i in range(len(evals)):
        evals[i]["metrics"] = get_eval_metrics(evals[i]["metrics"])

    lrlocals = []
    lrlocal_array = []
    for i in range(encode_num):
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
    for i in range(encode_num):
        fname = "Style_"+str(i)
        if inx!=-1:
            fdir = "evaluate"+"/"+str(inx)+"/"
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

    '''x_data = ("style_0", "style_1", "style_2")
    colors = ["SkyBlue", "IndianRed", "yellow"]

    for j in range(lenth):
        fig, ax = plt.subplots()
        for i in range(encode_num):
            ax.bar(i*0.5, [lrlocals[i][j]], 0.5, color=colors[i],label=x_data[i])
        ax.set_title(ep_stats[j])
        plt.xticks(np.arange(encode_num)*0.5, x_data)
        plt.savefig("./images/"+ep_stats[j]+".png")
        plt.close()'''

    #save expert data
    if save and not pi:
        idx = np.arange(len(obs_list))
        np.random.shuffle(idx)
        obs_list = np.array(obs_list)[idx]
        acs_list = np.array(acs_list)[idx]
        filename = env.spec.id
        np.savez(filename, obs=np.array(obs_list), acs=np.array(acs_list))


def traj_segment_sampler(env, path_num, encode_num=2, pi=None, stochastic=False):

    encode_axis = 0
    done_num = [0]*encode_num
    obs_list = []
    acs_list = []
    ep_true_rets, metrics = [],[]
    for i in range(encode_num):
        ep_true_rets.append([])
        metrics.append([])

    for i in tqdm(range(path_num)):
        if pi:
            encode = np.zeros((encode_num,), dtype=np.float32)
            encode[encode_axis] = 1
        else:
            env.set_ego_driving_style(encode_axis)

        ac = env.action_space.sample()
        ob_vars = env.reset()
        ob = env.ego_vehicle.get_ob_features(env.vehicle_list_separ_lanes)
        # Initialize history arrays
        obs = []
        true_rews = []
        acs = []

        while True:
            if pi:
                ac, _, _, _ = pi.act(stochastic, ob, encode)
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
                metrics[encode_axis].append(agent_info['episode']['metric'])
                if agent_info['episode']['l'] and agent_info['episode']['r']:
                    done_num[encode_axis] += 1
                    print(agent_info['episode'])
                    ep_true_rets[encode_axis].append(agent_info['episode']['r'])

                obs_list.append(np.array(obs))
                acs_list.append(np.array(acs))
                encode_axis = (encode_axis + 1) % encode_num
                break

    evals = []
    for i in range(encode_num):
        evals.append({"done_ratio": done_num[i]*encode_num/path_num, "ep_rets": ep_true_rets[i], "metrics": metrics[i]})

    return obs_list, acs_list, evals


if __name__ == '__main__':
    args = argsparser()
    main(args)