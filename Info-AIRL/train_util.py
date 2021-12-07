import numpy as np
from contextlib import contextmanager
from baselines.common import colorize
import time
from mpi4py import MPI
import gym
import matplotlib.pyplot as plt

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


def traj_segment_generator(pi, env, path_num, posterior_target, stochastic, encode_num=3):

    done_num = [0]*encode_num
    encode_axis = 0
    paths = []
    ep_true_rets, post_rets, metrics = [],[],[]
    for i in range(encode_num):
        ep_true_rets.append([])
        post_rets.append([])
        metrics.append([])

    for i in range(path_num):
        encode = np.zeros((encode_num, ), dtype=np.float32)
        encode[encode_axis] = 1
        encodes = []
        ac = env.action_space.sample()
        new = True
        ob_vars = env.reset()
        ob = env.ego_vehicle.get_ob_features(env.vehicle_list_separ_lanes)
        # Initialize history arrays
        obs = []
        true_rews = []
        sparse_rews = []
        news = []
        vpreds = []
        acs = []
        means = []
        logstds = []
        prevacs = []
        cur_post_ret = 0

        while True:
            prevac = ac
            ac, mean, logstd, vpred = pi.act(stochastic, ob, encode)
            cur_post_ret += posterior_target.get_reward(np.array([ob]), np.array([ac]), np.array([encode]))
            obs.append(ob)
            news.append(new)
            acs.append(ac)
            encodes.append(encode)
            vpreds.append(vpred)
            prevacs.append(prevac)
            if isinstance(env.action_space, gym.spaces.Box):
                means.append(mean)
                logstds.append(logstds)
            else:
                means.append(mean[ac])
                logstds.append(logstd[ac])
            ob_vars, true_rew, ego_done, agent_info = env.step([ac])
            ob = env.ego_vehicle.get_ob_features(ob_vars)
            true_rews.append(true_rew)
            sparse_rews.append(agent_info["aug_rwd"])
            new = agent_info['break']
            if new:
                post_rets[encode_axis].append(cur_post_ret)
                metrics[encode_axis].append(agent_info['episode']['metric'])
                if agent_info['episode']['l'] and agent_info['episode']['r']:
                    done_num[encode_axis] += 1
                    print(agent_info['episode'])
                    ep_true_rets[encode_axis].append(agent_info['episode']['r'])
                    #metrics[encode_axis].append(agent_info['episode']['metric'])
                path = {"ob": np.array(obs), "new": np.array(news), "ac": np.array(acs), "mean": np.array(means), "logstd": np.array(logstds), "vpred": np.array(vpreds),
                 "true_reward": np.array(true_rews), "sparse_reward": np.array(sparse_rews), "nextvpred": vpred * (1 - new), "prevac": np.array(prevacs), "encode": np.array(encodes)}
                paths.append(path)
                encode_axis = (encode_axis + 1) % encode_num
                break

    evals = []
    for i in range(encode_num):
        evals.append({"done_ratio": done_num[i]*encode_num/path_num, "ep_rets": ep_true_rets[i], "post_rets": post_rets[i], "metrics": metrics[i]})

    return paths, evals

def list_dict(l):
    d = {}
    for elem in l:
        for k, v in elem.items():
            if v:
                d.setdefault(k, []).append(v)
    key_list = l[0].keys()
    d_key_list = d.keys()
    for k in key_list:
        if k not in d_key_list:
            d[k]=[]
    return d

def get_eval_metrics(metric):
    dicts = list_dict(metric)
    dict_steps = list_dict(dicts["steps"])
    dict_acces = list_dict(dicts["acces"])
    dict_speeds = list_dict(dicts["speed"])
    dict_times = list_dict(dicts["time"])
    dict_gaps = list_dict(dicts["gaps"])

    return {"steps":dict_steps, "acces": dict_acces, "speeds": dict_speeds, "times": dict_times, "gaps": dict_gaps}

def extract_num(lst):
    ret_lst=[]
    for ele in lst:
        ret_lst.append(ele['gap2'])
    return ret_lst


def compute_path_probs(acs, means, logstds, ac_space, is_expert):

    horizon = len(acs)


    if isinstance(ac_space, gym.spaces.Discrete) or isinstance(ac_space, gym.spaces.MultiDiscrete):

        if is_expert:
            path_probs = [[0.0] for _ in range(horizon)]

        else:
            path_probs = [[np.log(means[i])] for i in range(horizon)]

    else:

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