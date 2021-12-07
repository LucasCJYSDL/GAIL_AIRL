import numpy as np
from contextlib import contextmanager
from baselines.common import colorize
import time
from mpi4py import MPI
import gym
import matplotlib.pyplot as plt

# def gauss_log_pdf(params, x):
#     mean, log_diag_std = params
#     d = mean.shape[0]
#     cov =  np.square(np.exp(log_diag_std))
#     diff = x-mean
#     exp_term = -0.5 * np.sum(np.square(diff)/cov, axis=0)
#     norm_term = -0.5*d*np.log(2*np.pi)
#     var_term = -0.5 * np.sum(np.log(cov), axis=0)
#     log_probs = norm_term + var_term + exp_term
#
#     return log_probs


def traj_segment_generator(pi, env, horizon, stochastic):

    # Initialize state variables

    t = 0
    epi_num = 0 #number of episodes in a trajectories
    done_num = 0 #number of episodes that are done successfully

    ac = env.action_space.sample()
    mean = ac
    logstd = np.zeros_like(np.array(mean))
    new = True
    true_rew = 0.0
    ob_vars = env.reset()
    ob = env.ego_vehicle.get_ob_features(env.vehicle_list_separ_lanes)

    cur_new_ret = 0
    cur_new_len = 0
    ep_true_rets = []
    ep_lens = []
    new_true_rets = []
    new_lens = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    true_rews = np.zeros(horizon, 'float32')
    sparse_rews = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')

    vpreds = np.zeros(horizon, 'float32')
    acs = np.array([ac for _ in range(horizon)])
    means = np.array([float(mean) for _ in range(horizon)])
    logstds = np.array([logstd for _ in range(horizon)])

    prevacs = acs.copy()

    while True:
        prevac = ac

        ac, mean, logstd, vpred = pi.act(stochastic, ob)

        if t > 0 and t % horizon == 0:

            yield {"ob": obs, "new": news, "ac": acs, "mean": means, "logstd": logstds, "vpred": vpreds, "true_reward": true_rews, "sparse_reward": sparse_rews, "new_lens": new_lens, "new_true_rets": new_true_rets,
                   "nextvpred": vpred * (1 - new), "prevac": prevacs, "ep_lens": ep_lens, "ep_true_rets": ep_true_rets, "done_ratio": done_num/epi_num}
            _, _, _, vpred = pi.act(stochastic, ob)
            ep_true_rets = []
            ep_lens = []
            new_true_rets = []
            new_lens = []
            epi_num = done_num = 0

        i = t % horizon
        obs[i] = ob
        news[i] = new
        acs[i] = ac
        vpreds[i] = vpred
        prevacs[i] = prevac

        if isinstance(env.action_space, gym.spaces.Box):
            means[i] = mean
            logstds[i] = logstd
        else:
            means[i] = mean[ac]
            logstds[i] = logstd[ac]

        ob_vars, true_rew, ego_done, agent_info = env.step([ac])
        ob = env.ego_vehicle.get_ob_features(ob_vars)

        true_rews[i] = true_rew
        cur_new_ret += true_rew

        sparse_rews[i] = agent_info["aug_rwd"]

        if true_rew != 0:
            cur_new_len += 1

        new = agent_info['break']
        if new:

            epi_num += 1
            new_true_rets.append(cur_new_ret)
            new_lens.append(cur_new_len)
            cur_new_ret = 0
            cur_new_len = 0
            if agent_info['episode']['l'] and agent_info['episode']['r']:

                done_num += 1

                print(agent_info['episode'])
                ep_true_rets.append(agent_info['episode']['r'])
                ep_lens.append(agent_info['episode']['l'])

            ob_vars = env.reset()
            ob = env.ego_vehicle.get_ob_features(env.vehicle_list_separ_lanes)
        t += 1

# def compute_path_probs(acs, means, logstds, ac_space, is_expert):
#
#     horizon = len(acs)
#
#
#     if isinstance(ac_space, gym.spaces.Discrete) or isinstance(ac_space, gym.spaces.MultiDiscrete):
#
#         if is_expert:
#             path_probs = [[0.0] for _ in range(horizon)]
#
#         else:
#             path_probs = [[np.log(means[i])] for i in range(horizon)]
#
#     else:
#
#         params = [(means[i], logstds[i]) for i in range(horizon)]
#         path_probs = [[gauss_log_pdf(params[i], acs[i])] for i in range(horizon)]
#
#     return np.array(path_probs)


def add_vtarg_and_adv(seg, gamma, lam):
    new = np.append(seg["new"], 0)  # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])

    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]

        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

@contextmanager
def timed(msg, rank):
    if rank == 0:
        print(colorize(msg, color='magenta'))
        tstart = time.time()
        yield
        print(colorize("done in %.3f seconds" % (time.time() - tstart), color='magenta'))
    else:
        yield

def allmean(x, nworkers):
    assert isinstance(x, np.ndarray)
    out = np.empty_like(x)
    MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
    out /= nworkers
    return out
