import numpy as np
from contextlib import contextmanager
from baselines.common import colorize
from mpi4py import MPI
import time

def traj_segment_generator(pi, env, reward_giver, horizon, stochastic):
    # Initialize state variables
    t = 0
    ac = env.action_space.sample()
    new = True
    rew = 0.0
    true_rew = 0.0
    ob = env.reset()

    cur_ep_ret = 0
    cur_ep_len = 0
    cur_ep_true_ret = 0
    ep_true_rets = []
    ep_rets = []
    ep_lens = []

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
            yield {"ob": obs, "rew": rews, "vpred": vpreds, "new": news,
                   "ac": acs, "prevac": prevacs, "nextvpred": vpred * (1 - new),#？？
                   "ep_rets": ep_rets, "ep_lens": ep_lens, "ep_true_rets": ep_true_rets}

            _, vpred = pi.act(stochastic, ob)
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_true_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        rew = reward_giver.get_reward(ob, ac)#真reward 假reward
        ob, true_rew, new, _ = env.step(ac)
        rews[i] = rew
        true_rews[i] = true_rew

        cur_ep_ret += rew
        cur_ep_true_ret += true_rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_true_rets.append(cur_ep_true_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_true_ret = 0
            cur_ep_len = 0
            ob = env.reset()
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