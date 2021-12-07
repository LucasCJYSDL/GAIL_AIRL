import numpy as np
from contextlib import contextmanager
from baselines.common import colorize
from mpi4py import MPI
import time
import matplotlib.pyplot as plt

def traj_segment_generator(pi, env, reward_giver, horizon, stochastic):
    # Initialize state variables
    t = 0

    epi_num = 0
    done_num = 0

    figsize_tuple = (18, 9)
    fig, ax = plt.subplots(1, figsize=figsize_tuple)

    ac = env.action_space.sample()
    new = True
    rew = 0.0
    true_rew = 0.0
    ob_vars = env.reset()
    ob = env.ego_vehicle.get_ob_features(env.vehicle_list_separ_lanes)

    cur_ep_ret = 0
    cur_new_ret = 0
    cur_new_len = 0
    ep_true_rets = []
    ep_lens = []
    ep_rets = []
    new_true_rets = []
    new_lens = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    true_rews = np.zeros(horizon, 'float32')
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob": obs, "rew": rews, "vpred": vpreds, "new": news, "done_ratio": done_num/epi_num,
                   "ac": acs, "prevac": prevacs, "nextvpred": vpred * (1 - new), "new_lens": new_lens, "new_true_rets": new_true_rets,
                   "ep_rets": ep_rets, "ep_lens": ep_lens, "ep_true_rets": ep_true_rets}

            _, vpred = pi.act(stochastic, ob)
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_true_rets = []
            ep_lens = []
            new_true_rets = []
            new_lens = []
            epi_num = done_num = 0

        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        rew = reward_giver.get_reward(ob, np.array([ac]))#真reward 假reward
        ob_vars, true_rew, ego_done, agent_info = env.step([ac])
        ob = env.ego_vehicle.get_ob_features(ob_vars)

        rews[i] = rew
        true_rews[i] = true_rew

        cur_ep_ret += rew

        cur_new_ret += true_rew

        if true_rew != 0:
            cur_new_len += 1

        new =agent_info['break']
        if new:
            ep_rets.append(cur_ep_ret)
            epi_num += 1
            new_true_rets.append(cur_new_ret)
            new_lens.append(cur_new_len)
            cur_new_ret = 0
            cur_new_len = 0
            cur_ep_ret = 0

            if agent_info['episode']['l'] and agent_info['episode']['r']:
                done_num += 1

                print(agent_info['episode'])
                ep_true_rets.append(agent_info['episode']['r'])
                ep_lens.append(agent_info['episode']['l'])

            ob_vars = env.reset()
            ob = env.ego_vehicle.get_ob_features(env.vehicle_list_separ_lanes)
        t += 1

def add_vtarg_and_adv(seg, gamma, lam):##完全一样
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