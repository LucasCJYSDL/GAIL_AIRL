import numpy as np
from contextlib import contextmanager
from baselines.common import colorize
import time
from mpi4py import MPI


def gauss_log_pdf(params, x):
    mean, log_diag_std = params
    d = mean.shape[0]
    cov =  np.square(np.exp(log_diag_std))
    diff = x-mean
    exp_term = -0.5 * np.sum(np.square(diff)/cov, axis=0)
    norm_term = -0.5*d*np.log(2*np.pi)
    var_term = -0.5 * np.sum(np.log(cov), axis=0)
    log_probs = norm_term + var_term + exp_term

    return log_probs


def traj_segment_generator(pi, env, horizon, stochastic):

    # Initialize state variables
    t = 0
    ac = env.action_space.sample()
    mean = ac
    logstd = np.array([0, 0, 0])
    new = True
    true_rew = 0.0
    ob = env.reset()

    cur_ep_len = 0
    cur_ep_true_ret = 0
    ep_true_rets = []
    ep_lens = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    true_rews = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')

    vpreds = np.zeros(horizon, 'float32')
    acs = np.array([ac for _ in range(horizon)])
    means = np.array([mean for _ in range(horizon)])
    logstds = np.array([logstd for _ in range(horizon)])

    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, mean, logstd, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:

            yield {"ob": obs, "new": news, "ac": acs, "mean": means, "logstd": logstds, "vpred": vpreds, "true_reward": true_rews,
                   "nextvpred": vpred * (1 - new), "prevac": prevacs, "ep_lens": ep_lens, "ep_true_rets": ep_true_rets}
            _, _, _, vpred = pi.act(stochastic, ob)
            ep_true_rets = []
            ep_lens = []

        i = t % horizon
        obs[i] = ob
        news[i] = new
        acs[i] = ac
        means[i] = mean
        logstds[i] = logstd
        vpreds[i] = vpred
        prevacs[i] = prevac

        ob, true_rew, new, _ = env.step(ac)
        true_rews[i] = true_rew

        cur_ep_true_ret += true_rew
        cur_ep_len += 1
        if new:
            ep_true_rets.append(cur_ep_true_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_true_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1

def compute_path_probs(acs, means, logstds):

    horizon = len(acs)
    params = [(means[i], logstds[i]) for i in range(horizon)]
    path_probs = [[gauss_log_pdf(params[i], acs[i])] for i in range(horizon)]

    return np.array(path_probs)


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