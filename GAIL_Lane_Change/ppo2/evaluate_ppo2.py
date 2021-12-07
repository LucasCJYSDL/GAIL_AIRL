from common.policies import build_policy
from common_util.runner import Runner
from tqdm import tqdm
from collections import deque
import numpy as np

def eval(*, network, env, reward_giver,nsteps, traj_num,
          ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, gamma=0.99, lam=0.95,
          nminibatches, value_network, load_model_path=None, model_fn=None, mpi_rank_weight=1, comm=None, num_hidden, num_layers, num_rnn_hidden):

    if network == "lstm":
        policy = build_policy(env, network, value_network=value_network, normalize_observations=True, nlstm = num_rnn_hidden)
    else:
        policy = build_policy(env, network, value_network=value_network, normalize_observations=True, num_layers= num_layers, num_hidden=num_hidden)

    nenvs = env.num_envs

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    # Calculate the batch_size
    nbatch = nsteps
    nbatch_train = nbatch // nminibatches

    # Instantiate the model object (that creates act_model and train_model)
    if model_fn is None:
        from ppo2.model import Model
        model_fn = Model

    model = model_fn(policy=policy, reward_giver=reward_giver, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs,
                     nbatch_train=nbatch_train, nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                     max_grad_norm=max_grad_norm, comm=comm, mpi_rank_weight=mpi_rank_weight, checkpoint_dir=load_model_path)

    if load_model_path is not None:
        model.load(load_model_path)
        print("Load successfully!")

    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam, reward_giver=reward_giver)

    obs_list = []
    acs_list = []
    epinfobuf = deque(maxlen=100)

    for _ in tqdm(range(traj_num)):
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run()
        obs_list.append(obs)
        acs_list.append(actions)
        epinfobuf.extend(epinfos)
    avg_len = safemean([epinfo['l'] for epinfo in epinfobuf])
    avg_ret = safemean([epinfo['r'] for epinfo in epinfobuf])
    print("Average length:", avg_len)
    print("Average return:", avg_ret)
    return avg_len, avg_ret

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)