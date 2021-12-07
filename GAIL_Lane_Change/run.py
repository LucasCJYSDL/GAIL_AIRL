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
import os
from common_util import mlp_policy
from baselines.common import set_global_seeds
from common import tf_util as U
from baselines.common.misc_util import boolean_flag
from baselines import bench
from baselines import logger
from dataset.dset import AutoDrive_Dset
from common_util.adversary import TransitionClassifier
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
import gymAutoDrive


def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of GAIL")
    parser.add_argument('--env_id', help='environment ID', default='AutoDrive-v0')#game
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)#随机种子
    parser.add_argument('--expert_path', type=str, default='./data/AutoDrive-v0-5000.npz')#专家数据
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='checkpoint')#检查点
    parser.add_argument('--log_dir', help='the directory to save log file', default='log')#日志
    parser.add_argument('--load_model_path_ind', help='if provided, load the model index', type=int, default=-1)#预训练模型
    # Task
    parser.add_argument('--task', type=str, choices=['train', 'evaluate'], default='train')#执行的任务，sample是啥
    # for evaluatation
    boolean_flag(parser, 'stochastic_policy', default=False, help='use stochastic/deterministic policy to evaluate')#是否用随机策略，随机策略指啥
    boolean_flag(parser, 'save_sample', default=False, help='save the trajectories or not')#是否存储轨迹
    #  Mujoco Dataset Configuration
    parser.add_argument('--traj_limitation', type=int, default=-1)#生成轨迹长度？
    # Optimization Configuration
    parser.add_argument('--g_step', help='number of steps to train policy in each epoch', type=int, default=14)
    parser.add_argument('--d_step', help='number of steps to train discriminator in each epoch', type=int, default=1)#generator迭代6次，discriminator迭代1次
    # Network Configuration (Using MLP Policy)
    parser.add_argument('--policy_hidden_size', type=int, default=100)#隐含层的节点数？
    parser.add_argument('--adversary_hidden_size', type=int, default=512)#？？？
    parser.add_argument('--rnn_hidden_size', type=int, default=128)
    # Algorithms Configuration
    parser.add_argument('--algo', type=str, choices=['trpo', 'ppo', 'ppo2'], default='trpo')
    parser.add_argument('--pi_nn', type=str, choices=['mlp','lstm','gru','mlpgru'], default='mlp')
    parser.add_argument('--max_kl', type=float, default=0.01)
    parser.add_argument('--policy_entcoeff', help='entropy coefficiency of policy', type=float, default=0)#loss函数中某一项前的系数
    parser.add_argument('--adversary_entcoeff', help='entropy coefficiency of discriminator', type=float, default=1e-3)
    # Traing Configuration
    parser.add_argument('--save_per_iter', help='save model every xx iterations', type=int, default=100)#迭代多少次存储检查点
    parser.add_argument('--num_timesteps', help='number of timesteps per episode', type=int, default=1e7)#训练步长
    # Behavior Cloning
    boolean_flag(parser, 'pretrained', default=False, help='Use BC to pretrain')#是否用BC预训练
    parser.add_argument('--BC_max_iter', help='Max iteration for training BC', type=int, default=1e4)#使用BC预训练的最大迭代次数
    return parser.parse_args()


def get_task_name(args):
    task_name = args.algo + '_' + args.pi_nn
    task_name = task_name + "_gail."
    if args.pretrained:
        task_name += "with_pretrained."
    if args.traj_limitation != np.inf:
        task_name += "transition_limitation_%d." % args.traj_limitation
    task_name += args.env_id.split("-")[0]
    if args.algo != 'mlp':
        task_name = task_name + ".rnn_size_" + str(args.rnn_hidden_size)
    task_name = task_name + ".g_step_" + str(args.g_step) + ".d_step_" + str(args.d_step) + \
        ".policy_entcoeff_" + str(args.policy_entcoeff) + ".adversary_entcoeff_" + str(args.adversary_entcoeff)
    task_name += ".seed_" + str(args.seed)
    return task_name #记录方便

def get_load_model_path(args, ind):
    if ind == -1:
        return None
    load_model_path = args.checkpoint_dir + '/' + get_task_name(args) + str(ind)
    return load_model_path


def main(args):
    U.make_session(num_cpu=1).__enter__()#指定设备创立并进入session
    set_global_seeds(args.seed)#设置随机种子
    env = gym.make(args.env_id)#启动交互环境

    anim_episode = 1
    env.init_global_parameters_for_gather_data(anim_episode)

    def policy_fn(type, name, ob_space, ac_space, nbatch=0, reuse=False):
        if type == 'mlp':
            return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    reuse=reuse, hid_size=args.policy_hidden_size, num_hid_layers=2)#输入输出层节点数，隐含层个数，隐含层节点数
        elif type == 'mlpgru':
            return mlp_policy.MlpGruPolicy(name = name, ob_space = ob_space, ac_space = ac_space, nbatch = nbatch,
                                           reuse=reuse, ngru=args.rnn_hidden_size, hid_size=args.policy_hidden_size, num_hid_layers=2)
        elif type == 'gru':
            return mlp_policy.GruPolicy(name=name, ob_space=ob_space, ac_space=ac_space, nbatch = nbatch, reuse=reuse, ngru=args.rnn_hidden_size)

        elif type == 'lstm':
            return mlp_policy.LstmPolicy(name=name, ob_space=ob_space, ac_space=ac_space, nbatch=nbatch, reuse=reuse, nlstm=args.rnn_hidden_size)

    #env = bench.Monitor(env, logger.get_dir() and osp.join(logger.get_dir(), "monitor.json"))
    env.seed(args.seed)
    gym.logger.setLevel(logging.WARN)
    task_name = get_task_name(args)
    args.checkpoint_dir = osp.join(args.checkpoint_dir, task_name)
    args.log_dir = osp.join(args.log_dir, task_name)
    reward_giver = TransitionClassifier(env, args.adversary_hidden_size, entcoeff=args.adversary_entcoeff)

    if args.task == 'train':
        #dataset = Mujoco_Dset(expert_path=args.expert_path, traj_limitation=args.traj_limitation)
        dataset = AutoDrive_Dset(expert_path=args.expert_path, traj_limitation=args.traj_limitation, randomize = (args.pi_nn == "mlp"))
        train(env,
              args.seed,
              args.pi_nn,
              policy_fn,
              reward_giver,
              dataset,
              args.algo,
              args.g_step,
              args.d_step,
              args.policy_entcoeff,
              args.num_timesteps,
              args.save_per_iter,
              args.checkpoint_dir,
              args.log_dir,
              args.pretrained,
              args.BC_max_iter,
              task_name
              )
    elif args.task == 'evaluate':
        if args.algo == 'ppo2':
            from ppo2.evaluate_ppo2 import eval
            env_ref = DummyVecEnv([lambda: env])
            checkpoints = []
            for file in os.listdir(args.checkpoint_dir):
                checkpoints.append(file)
            clen = len(checkpoints)
            load_model_path = osp.join(args.checkpoint_dir, checkpoints[clen-1])
            print(load_model_path)
            eval(  env=env_ref, network=args.pi_nn, reward_giver=reward_giver, traj_num = 10, load_model_path = load_model_path,
                   nsteps=1024, nminibatches=16, lam=0.95, gamma=0.99, ent_coef=0.0, vf_coef=1.0,  # 0.5 or 1.0
                   value_network=None, max_grad_norm=0.5, num_hidden=args.policy_hidden_size, num_layers=2, num_rnn_hidden = args.rnn_hidden_size)

        else:
            if args.pi_nn == "mlp":
                env_ref = env
            else:
                env_ref = DummyVecEnv([lambda: env])
            runner(env_ref,
                   policy_fn,
                   get_load_model_path(args, args.load_model_path_ind),
                   stochastic=args.stochastic_policy,
                   pi_nn=args.pi_nn,
                   reward_giver=reward_giver,
                   timesteps_per_batch=1024,
                   number_trajs=10,
                   save=args.save_sample,
                   lam=0.95, gamma=0.99
                   )
    else:
        raise NotImplementedError
    env.close()


def train(env, seed, pi_nn, policy_fn, reward_giver, dataset, algo,
          g_step, d_step, policy_entcoeff, num_timesteps, save_per_iter,
          checkpoint_dir, log_dir, pretrained, BC_max_iter, task_name=None):

    pretrained_weight = None
    if pretrained and (BC_max_iter > 0):###9
        # Pretrain with behavior cloning
        from common_util import behavior_clone
        pretrained_weight = behavior_clone.learn(env, policy_fn, dataset, max_iters=BC_max_iter)

    if algo == 'trpo':
        from trpo import trpo_mpi
        # Set up for MPI seed
        rank = MPI.COMM_WORLD.Get_rank()
        if rank != 0:
            logger.set_level(logger.DISABLED)
        workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
        set_global_seeds(workerseed)
        env.seed(workerseed)
        if pi_nn == "mlp":
            env_ref = env
        else:
            env_ref = DummyVecEnv([lambda: env])
        trpo_mpi.learn(env_ref, policy_fn, reward_giver, dataset, rank, pi_nn = pi_nn,
                       pretrained=pretrained, pretrained_weight=pretrained_weight,
                       g_step=g_step, d_step=d_step,
                       entcoeff=policy_entcoeff,
                       max_timesteps=num_timesteps, max_grad_norm = 0.5,
                       ckpt_dir=checkpoint_dir, log_dir=log_dir,
                       save_per_iter=save_per_iter,
                       timesteps_per_batch=1024, optim_batchsize=1024,
                       max_kl=args.max_kl, cg_iters=10, cg_damping=0.1,
                       gamma=0.995, lam=0.97, schedule='linear',
                       vf_iters=5, vf_stepsize=1e-3,
                       task_name=task_name, load_model_path = get_load_model_path(args, args.load_model_path_ind))

    elif algo == 'ppo':
        from ppo import ppo_mpi
        rank = MPI.COMM_WORLD.Get_rank()
        if rank != 0:
            logger.set_level(logger.DISABLED)
        workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
        set_global_seeds(workerseed)
        env.seed(workerseed)
        if pi_nn == "mlp":
            env_ref = env
        else:
            env_ref = DummyVecEnv([lambda: env])
        ppo_mpi.learn(env_ref, policy_fn, reward_giver, dataset, rank, pi_nn = pi_nn,
                      pretrained=pretrained, pretrained_weight=pretrained_weight,
                      d_step = d_step,
                      clip_param = 0.2, entcoeff = policy_entcoeff,
                      ckpt_dir=checkpoint_dir, log_dir=log_dir,
                      max_timesteps=num_timesteps, max_grad_norm = 0.5,
                      save_per_iter=save_per_iter, timesteps_per_actorbatch = 1024,
                      optim_epochs=g_step, optim_stepsize=3e-4, optim_batchsize=64,
                      gamma=0.99, lam=0.95, schedule='linear', task_name=task_name,
                      load_model_path = get_load_model_path(args, args.load_model_path_ind))

    elif algo == 'ppo2':
        from ppo2 import ppo2_mpi
        rank = MPI.COMM_WORLD.Get_rank()
        if rank != 0:
            logger.set_level(logger.DISABLED)
        workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
        env.seed(workerseed)
        env_ref = DummyVecEnv([lambda: env])
        ppo2_mpi.learn(env=env_ref, network='lstm', reward_giver=reward_giver, expert_dataset=dataset, total_timesteps=num_timesteps,
                   nsteps=1024, nminibatches=16, lam=0.95, gamma=0.99, seed = workerseed, log_dir = log_dir,
                   noptepochs=g_step, log_interval=1, ent_coef=0.0, d_step = d_step, vf_coef=1.0,#0.5 or 1.0
                   lr=lambda f: 3e-4 * f, cliprange=0.2, value_network=None, max_grad_norm=0.5, d_stepsize=3e-4,
                   save_interval=save_per_iter, load_model_path=get_load_model_path(args, args.load_model_path_ind), num_hidden=args.policy_hidden_size, num_layers=2, num_rnn_hidden = args.rnn_hidden_size)

    else:
        raise NotImplementedError


def runner(env, policy_func, load_model_path, stochastic, pi_nn, reward_giver, timesteps_per_batch, number_trajs,
           gamma, lam, save=False, reuse=False):

    from common_util.runner import Runner
    from collections import deque

    # Setup network
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space

    if pi_nn == "mlp":
        pi = policy_func(pi_nn, "pi", ob_space, ac_space, reuse=reuse)
    else:
        pi = policy_func(pi_nn, "pi", ob_space, ac_space, 1, reuse=reuse)
        runner = Runner(env=env, model=pi, pi_nn=pi_nn, nsteps=timesteps_per_batch, gamma=gamma, lam=lam, reward_giver=reward_giver)

    U.initialize()
    # Prepare for rollouts
    # ----------------------------------------

    U.load_state(load_model_path)

    obs_list = []
    acs_list = []
    if pi_nn == "mlp":
        len_list = []
        ret_list = []
    else:
        epinfobuf = deque(maxlen=100)

    for _ in tqdm(range(number_trajs)):
        if pi_nn == "mlp":
            traj = traj_1_generator(pi, env, timesteps_per_batch, stochastic=stochastic)
            ob, ac, ep_len, ep_ret = traj['ob'], traj['ac'], traj['ep_len'], traj['ep_ret']
            len_list.append(ep_len)
            ret_list.append(ep_ret)
        elif pi_nn == "lstm" or pi_nn == "gru":
            ob, tdlamret, masks, ac, values, logps, state, epinfos = runner.run()
            epinfobuf.extend(epinfos)
        elif pi_nn == "mlpgru":
            ob, tdlamret, masks, ac, values, logps, _, state, epinfos = runner.run()
            epinfobuf.extend(epinfos)
        obs_list.append(ob)
        acs_list.append(ac)

    if save:
        filename = load_model_path.split('/')[-1] + '.' + env.spec.id
        np.savez(filename, obs=np.array(obs_list), acs=np.array(acs_list))

    if pi_nn == "mlp":
        avg_len = sum(len_list) / len(len_list)
        avg_ret = sum(ret_list) / len(ret_list)
    else:
        avg_len = safemean([epinfo['l'] for epinfo in epinfobuf])
        avg_ret = safemean([epinfo['r'] for epinfo in epinfobuf])

    print("Average length:", avg_len)
    print("Average return:", avg_ret)
    return avg_len, avg_ret


def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

# Sample one trajectory (until trajectory end)
def traj_1_generator(pi, env, horizon, stochastic):

    t = 0
    ac = env.action_space.sample()  # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode

    ob = env.reset()
    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode

    # Initialize history arrays
    obs = []
    rews = []
    news = []
    acs = []

    while True:
        ac, vpred = pi.act(stochastic, ob)
        obs.append(ob)
        news.append(new)
        acs.append(ac)

        ob, rew, new, _ = env.step(ac)
        rews.append(rew)

        cur_ep_ret += rew
        cur_ep_len += 1
        if new or t >= horizon:
            break
        t += 1

    obs = np.array(obs)
    rews = np.array(rews)
    news = np.array(news)
    acs = np.array(acs)
    traj = {"ob": obs, "rew": rews, "new": news, "ac": acs,
            "ep_ret": cur_ep_ret, "ep_len": cur_ep_len}
    return traj


if __name__ == '__main__':
    args = argsparser()
    main(args)
