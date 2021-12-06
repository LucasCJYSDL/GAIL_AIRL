'''
Data structure of the input .npz:
the data is save in python dictionary format with keys: 'acs', 'ep_rets', 'rews', 'obs'
the values of each item is a list storing the expert trajectory sequentially
a transition can be: (data['obs'][t], data['acs'][t], data['obs'][t+1]) and get reward data['rews'][t]
'''

from baselines import logger
import numpy as np


class Dset(object):
    def __init__(self, inputs, labels_1, labels_2, labels_3, randomize):
        self.inputs = inputs

        self.labels_1 = labels_1
        self.labels_2 = labels_2
        self.labels_3 = labels_3

        assert len(self.inputs) == len(self.labels_1)
        assert len(self.inputs) == len(self.labels_2)
        assert len(self.inputs) == len(self.labels_3)

        self.randomize = randomize
        self.num_pairs = len(inputs)
        self.init_pointer()

    def init_pointer(self):
        self.pointer = 0
        if self.randomize:
            idx = np.arange(self.num_pairs)
            np.random.shuffle(idx)
            self.inputs = self.inputs[idx, :]
            self.labels_1 = self.labels_1[idx, :]
            self.labels_2 = self.labels_2[idx, :]
            self.labels_3 = self.labels_3[idx, :]

    def get_next_batch(self, batch_size):
        # if batch_size is negative -> return all
        if batch_size < 0:
            return self.inputs, self.labels_1, self.labels_2, self.labels_3
        if self.pointer + batch_size >= self.num_pairs:
            self.init_pointer()
        end = self.pointer + batch_size
        inputs = self.inputs[self.pointer:end, :]
        labels_1 = self.labels_1[self.pointer:end, :]
        labels_2 = self.labels_2[self.pointer:end, :]
        labels_3 = self.labels_3[self.pointer:end, :]
        self.pointer = end
        return inputs, labels_1, labels_2, labels_3


class Mujoco_Dset(object):
    def __init__(self, expert_path, train_fraction=0.7, traj_limitation=-1, randomize=True):
        traj_data = np.load(expert_path)
        if traj_limitation < 0:
            traj_limitation = len(traj_data['obs'])
        obs = traj_data['obs'][:traj_limitation]
        acs = traj_data['acs'][:traj_limitation]
        means = traj_data['means'][:traj_limitation]
        logstds = traj_data['logstds'][:traj_limitation]

        '''print(obs.shape)
        print(acs.shape)
        print(means.shape)
        print(logstds.shape)'''

        # obs, acs: shape (N, L, ) + S where N = # episodes, L = episode length
        # and S is the environment observation/action space.
        # Flatten to (N * L, prod(S))
        if len(obs.shape) > 2:
            self.obs = np.reshape(obs, [-1, np.prod(obs.shape[2:])])
            self.acs = np.reshape(acs, [-1, np.prod(acs.shape[2:])])
            self.means = np.reshape(means, [-1, np.prod(means.shape[2:])])
            self.logstds = np.reshape(logstds, [-1, np.prod(logstds.shape[2:])])
        else:
            self.obs = np.vstack(obs)
            self.acs = np.vstack(acs)
            self.means = np.vstack(means)
            self.logstds = np.vstack(logstds)

        self.rets = traj_data['rets'][:traj_limitation]
        self.avg_ret = sum(self.rets)/len(self.rets)
        self.std_ret = np.std(np.array(self.rets))
        if len(self.acs) > 2:
            self.acs = np.squeeze(self.acs)
        if len(self.means) > 2:
            self.means = np.squeeze(self.means)
        if len(self.logstds) > 2:
            self.logstds = np.squeeze(self.logstds)

        assert len(self.obs) == len(self.acs)
        assert len(self.obs) == len(self.means)
        assert len(self.obs) == len(self.logstds)

        self.num_traj = min(traj_limitation, len(traj_data['obs']))
        self.num_transition = len(self.obs)
        self.randomize = randomize
        self.dset = Dset(self.obs, self.acs, self.means, self.logstds, self.randomize)
        # for behavior cloning
        self.train_set = Dset(self.obs[:int(self.num_transition*train_fraction), :],
                              self.acs[:int(self.num_transition*train_fraction), :],
                              self.means[:int(self.num_transition*train_fraction), :],
                              self.logstds[:int(self.num_transition*train_fraction), :],
                              self.randomize)
        self.val_set = Dset(self.obs[int(self.num_transition*train_fraction):, :],
                            self.acs[int(self.num_transition*train_fraction):, :],
                            self.means[int(self.num_transition * train_fraction):, :],
                            self.logstds[int(self.num_transition * train_fraction):, :],
                            self.randomize)
        self.log_info()

    def log_info(self):
        logger.log("Total trajectorues: %d" % self.num_traj)
        logger.log("Total transitions: %d" % self.num_transition)
        logger.log("Average returns: %f" % self.avg_ret)
        logger.log("Std for returns: %f" % self.std_ret)

    def get_next_batch(self, batch_size, split=None):
        if split is None:
            return self.dset.get_next_batch(batch_size)
        elif split == 'train':
            return self.train_set.get_next_batch(batch_size)
        elif split == 'val':
            return self.val_set.get_next_batch(batch_size)
        else:
            raise NotImplementedError

    def plot(self):
        import matplotlib.pyplot as plt
        plt.hist(self.rets)
        plt.savefig("histogram_rets.png")
        plt.close()


def test(expert_path, traj_limitation, plot):
    dset = Mujoco_Dset(expert_path, traj_limitation=traj_limitation)
    if plot:
        dset.plot()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_path", type=str, default="../data/trpo_gail.transition_limitation_-1.Hopper.g_step_3.d_step_1.policy_entcoeff_0.adversary_entcoeff_0.001.seed_0.Hopper-v2.npz")
    parser.add_argument("--traj_limitation", type=int, default=-1)
    parser.add_argument("--plot", type=bool, default=False)
    args = parser.parse_args()
    test(args.expert_path, args.traj_limitation, args.plot)
