
'''
Disclaimer: this code is highly based on trpo_mpi at @openai/baselines and @openai/imitation
'''

import argparse
import os.path as osp
import logging
from mpi4py import MPI
from tqdm import tqdm
import numpy as np
import gym
import mlp_policy
from baselines.common import set_global_seeds, tf_util as U
from baselines.common.misc_util import boolean_flag
from baselines import logger
import gymAutoDrive
from dataset.dset import AutoDrive_Dset
import trpo_mpi

def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of TRPO")
    parser.add_argument('--env_id', help='environment ID', default='AutoDrive-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='checkpoint')
    parser.add_argument('--expert_path', type=str, default='./data/AutoDrive-v0-5000.npz')
    parser.add_argument('--log_dir', help='the directory to save log file', default='log')
    parser.add_argument('--load_model_ind', help='if positive, load the corresponding model', type=int, default=-1)
    # Task
    parser.add_argument('--task', type=str, choices=['train', 'evaluate'], default='train')
    # for evaluatation
    boolean_flag(parser, 'stochastic_policy', default=False, help='use stochastic/deterministic policy to evaluate')
    boolean_flag(parser, 'save_sample', default=True, help='save the trajectories or not')
    parser.add_argument('--num_of_trajs', type=int, default=1000)
    #  Mujoco Dataset Configuration
    parser.add_argument('--traj_limitation', type=int, default=-1)
    parser.add_argument('--exp_ind', type=str, default="0")
    # Network Configuration (Using MLP Policy)
    parser.add_argument('--policy_hidden_size', type=int, default=256)
    # Algorithms Configuration
    parser.add_argument('--max_kl', type=float, default=0.01)
    parser.add_argument('--policy_entcoeff', help='entropy coefficiency of policy', type=float, default=0.01)
    parser.add_argument('--use_default', help='whether to use the reward function defined in the paper(AIRL)',
                        default=True)
    # Traing Configuration
    parser.add_argument('--save_per_iter', help='save model every xx iterations', type=int, default=100)
    parser.add_argument('--num_timesteps', help='number of timesteps per episode', type=int, default=1e7)
    parser.add_argument('--use_true_reward',
                        help='whether to add the true reward(get from env.step) to the reward function', default=True)

    parser.add_argument('--use_sparse_reward',
                        help='whether to add the sparse reward(get from env.step) to the reward function', default=False)
    # Behavior Cloning
    boolean_flag(parser, 'pretrained', default=True, help='Use BC to pretrain')
    parser.add_argument('--BC_max_iter', help='Max iteration for training BC', type=int, default=3e3)
    return parser.parse_args()


def get_task_name(args):
    task_name = "TRPO_Lane_Change_" + args.exp_ind
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

    env.seed(args.seed)
    gym.logger.setLevel(logging.WARN)
    task_name = get_task_name(args)
    args.checkpoint_dir = osp.join(args.checkpoint_dir, task_name)
    args.log_dir = osp.join(args.log_dir, task_name)

    if args.task == 'train':

        dataset = AutoDrive_Dset(expert_path=args.expert_path, traj_limitation=args.traj_limitation)

        train(env,
              args.seed,
              policy_fn,
              args.policy_entcoeff,
              args.num_timesteps,
              args.save_per_iter,
              args.checkpoint_dir,
              args.log_dir,
              args.pretrained,
              args.BC_max_iter,
              get_load_path(args),
              args.use_true_reward,
              args.use_sparse_reward,
              dataset,
              task_name
              )
    elif args.task == 'evaluate':

        runner(env,
               policy_fn,
               get_load_path(args),
               number_trajs=args.num_of_trajs,
               stochastic_policy=args.stochastic_policy,
               save=args.save_sample
               )

    else:
        raise NotImplementedError
    env.close()


def train(env, seed, policy_fn, policy_entcoeff, num_timesteps, save_per_iter, checkpoint_dir, log_dir,
          pretrained, BC_max_iter, load_path, use_true_reward, use_sparse_reward, dataset, task_name=None):

    pretrained_weight = None
    if pretrained and (BC_max_iter > 0):
        # Pretrain with behavior cloning
        import behavior_clone
        pretrained_weight = behavior_clone.learn(env, policy_fn, dataset, task_name=task_name,
                                                 max_iters=BC_max_iter, ckpt_dir=checkpoint_dir)


    # Set up for MPI seed
    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    env.seed(workerseed)
    trpo_mpi.learn(env, policy_fn, dataset, rank,
                   pretrained=pretrained, pretrained_weight=pretrained_weight,
                   entcoeff=policy_entcoeff, max_timesteps=num_timesteps,
                   ckpt_dir=checkpoint_dir, log_dir=log_dir,
                   save_per_iter=save_per_iter,
                   timesteps_per_batch=1024,
                   max_kl=0.01, cg_iters=10, cg_damping=0.1, gamma=0.995, lam=0.97,
                   vf_iters=5, vf_stepsize=1e-3, load_path=load_path, use_true_reward=use_true_reward,
                   use_sparse_reward=use_sparse_reward, task_name=task_name)


def runner(env, policy_func, load_model_path, number_trajs, stochastic_policy, save=False, reuse=False):

    # Setup network
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space, reuse=reuse)

    if load_model_path:
        U.initialize()
        # Prepare for rollouts
        U.load_state(load_model_path)

    else:
        pi = None

    len_list = []
    ret_list = []
    new_lens = []
    obs_list = []
    acs_list = []
    done_num = 0  # number of episodes that are done successfully

    for i in tqdm(range(number_trajs)):

        traj = traj_1_generator(i, pi, env, stochastic=stochastic_policy)

        obs, acs, ep_len, ep_ret, new_len, new_ret = traj['ob'], traj['ac'], traj['ep_len'], traj['ep_ret'], traj['new_len'], traj['new_ret']

        obs_list.extend(obs)
        acs_list.extend(acs)

        if ep_len and ep_ret:
            done_num += 1
            len_list.append(ep_len)
            ret_list.append(ep_ret)

        new_lens.append(new_len)

    avg_len = sum(len_list) / len(len_list)
    avg_ret = sum(ret_list) / len(ret_list)
    done_ratio = done_num / number_trajs
    avg_new_len = sum(new_lens) / len(new_lens)

    len_array = np.array(len_list)
    ret_array = np.array(ret_list)
    new_len_array = np.array(new_lens)
    obs_array = np.array(obs_list)
    acs_array = np.array(acs_list)

    print("obs: ", obs_list, " shape: ", obs_array.shape)
    print("acs: ", acs_list, " shape: ", acs_array.shape)

    f_obs = open("driving_obs.txt", 'w')
    f_acs = open("driving_acs.txt", 'w')
    length = obs_array.shape[0]
    f_obs_head = [str(obs_array.shape[0]), str(obs_array.shape[1])]
    f_obs.write(f_obs_head[0] + " " + f_obs_head[1] + "\n")

    for i in range(length):
        for j in range(obs_array.shape[1] - 1):
            f_obs.write(str(obs_list[i][j]) + " ")
        f_obs.write(str(obs_list[i][obs_array.shape[1]-1]) + "\n")
        f_acs.writelines(str(acs_list[i][0]) + "\n")

    f_obs.close()
    f_acs.close()

    print("Average length:", avg_len)
    print("Average return:", avg_ret)
    print("Done ratio:", done_ratio)
    print("Average decision length:", avg_new_len)

    print("Length std:", np.std(len_array))
    print("Reward std:", np.std(ret_array))
    print("New length std:", np.std(new_len_array))

# Sample one trajectory (until trajectory end)

def traj_1_generator(i, pi, env, horizon, stochastic, is_visual=True):

    ac = env.action_space.sample()  # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode

    ob_vars = env.reset()
    ob = env.ego_vehicle.get_ob_features(env.vehicle_list_separ_lanes)

    cur_ep_ret = None  # return in current episode
    cur_ep_len = None  # len of current episode

    # Initialize history arrays
    obs = []
    rews = []
    news = []
    acs = []

    cur_new_ret = 0
    cur_new_len = 0

    while True:

        if pi:
            ac, _, _, _ = pi.act(stochastic, ob)
            ac = [ac]
        else:
            ac = env.temp_agent_act(ob)

        obs.append(ob)
        news.append(new)
        acs.append(ac)

        ob_vars, rew, ego_done, agent_info = env.step(ac)
        ob = env.ego_vehicle.get_ob_features(ob_vars)
        rews.append(rew)

        cur_new_ret += rew
        if rew!=0:
            cur_new_len += 1

        new = agent_info['break']

        if new:
            print(agent_info)
            print("last reward:")
            print(rew)

            if agent_info['episode']['l'] and agent_info['episode']['r']:
                cur_ep_len = agent_info['episode']['l']
                cur_ep_ret = agent_info['episode']['r']

            break

    # obs = np.array(obs)
    rews = np.array(rews)
    news = np.array(news)
    # acs = np.array(acs)

    traj = {"ob": obs, "rew": rews, "new": news, "ac": acs,
            "ep_ret": cur_ep_ret, "ep_len": cur_ep_len, "new_ret": cur_new_ret, "new_len": cur_new_len}

    return traj


if __name__ == '__main__':
    args = argsparser()
    main(args)