'''
Reference: https://github.com/openai/imitation
I follow the architecture from the official repository
'''
import tensorflow as tf
import numpy as np

from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common import tf_util as U

def logsigmoid(a):
    '''Equivalent to tf.log(tf.sigmoid(a))'''
    return -tf.nn.softplus(-a)

""" Reference: https://github.com/openai/imitation/blob/99fbccf3e060b6e6c739bdf209758620fcdefd3c/policyopt/thutil.py#L48-L51"""
def logit_bernoulli_entropy(logits):
    ent = (1.-tf.nn.sigmoid(logits))*logits - logsigmoid(logits)
    return ent

class TransitionClassifier(object):
    def __init__(self, env, hidden_size, use_default = False, entcoeff=0.001, lr_rate=1e-3, l2_reg = 0, scope="adversary"):
        self.scope = scope
        self.observation_shape = env.observation_space.shape
        self.actions_shape = env.action_space.shape
        self.input_shape = tuple([o+a for o, a in zip(self.observation_shape, self.actions_shape)])
        self.num_actions = env.action_space.shape[0]
        self.hidden_size = hidden_size
        self.build_ph()
        # Build grpah

        generator_logp = self.build_graph(self.generator_obs_ph, self.generator_acs_ph, reuse=False)
        expert_logp = self.build_graph(self.expert_obs_ph, self.expert_acs_ph, reuse=True)

        self.energy = tf.concat([generator_logp, expert_logp], axis=0)
        log_p = -self.energy
        log_q = self.lprobs_ph
        log_pq = tf.reduce_logsumexp([log_p, log_q], axis=0)
        self.d_tau = tf.exp(log_p - log_pq)
        cent_loss = -tf.reduce_mean(self.labels_ph * (log_p - log_pq) + (1 - self.labels_ph) * (log_q - log_pq))
        # Build regnum loss

        self.losses = [cent_loss]
        self.loss_name = ["cent_loss"]
        self.total_loss = cent_loss
        # Build Reward for policy

        self.debug = U.function(
            [self.generator_obs_ph, self.generator_acs_ph, self.expert_obs_ph, self.expert_acs_ph, self.labels_ph,
             self.lprobs_ph],
            (self.d_tau, log_p, log_q, tf.log(self.d_tau) - tf.log(1-self.d_tau)))

        if use_default:
            self.Reward, _reward_expert = tf.split(axis=0, num_or_size_splits=2, value=tf.log(self.d_tau + 1e-8) - tf.log(1-self.d_tau + 1e-8))
        else:
            self.Reward, _reward_expert = tf.split(axis=0, num_or_size_splits=2, value=self.d_tau)

        var_list = self.get_trainable_variables()

        self.lossandgrad = U.function([self.generator_obs_ph, self.generator_acs_ph, self.expert_obs_ph, self.expert_acs_ph, self.labels_ph, self.lprobs_ph],
                                      self.losses + [U.flatgrad(self.total_loss, var_list)])

    def build_ph(self):
        self.generator_obs_ph = tf.placeholder(tf.float32, (None, ) + self.observation_shape, name="observations_ph")
        self.generator_acs_ph = tf.placeholder(tf.float32, (None, ) + self.actions_shape, name="actions_ph")
        self.expert_obs_ph = tf.placeholder(tf.float32, (None, ) + self.observation_shape, name="expert_observations_ph")
        self.expert_acs_ph = tf.placeholder(tf.float32, (None, ) + self.actions_shape, name="expert_actions_ph")
        self.labels_ph = tf.placeholder(tf.float32, (None, 1), name='labels_ph')
        self.lprobs_ph = tf.placeholder(tf.float32, (None, 1), name='log_probs_ph')

    def build_graph(self, obs_ph, acs_ph, reuse=False):
        with tf.variable_scope(self.scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("obfilter"):
                self.obs_rms = RunningMeanStd(shape=self.observation_shape)
            obs = (obs_ph - self.obs_rms.mean) / self.obs_rms.std
            _input = tf.concat([obs, acs_ph], axis=1)  # concatenate the two input -> form a transition
            p_h1 = tf.contrib.layers.fully_connected(_input, self.hidden_size, activation_fn=tf.nn.relu)
            p_h2 = tf.contrib.layers.fully_connected(p_h1, self.hidden_size, activation_fn=tf.nn.relu)
            logits = tf.contrib.layers.fully_connected(p_h2, 1, activation_fn=tf.identity)
        return logits

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_reward(self, obs, acs, exp_obs, exp_acs, batch_size, lprobs):
        sess = tf.get_default_session()
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, 0)
        if len(acs.shape) == 1:
            acs = np.expand_dims(acs, 0)

        labels = np.zeros((batch_size * 2, 1))
        labels[batch_size:] = 1.0

        feed_dict = {self.generator_obs_ph: obs, self.generator_acs_ph: acs, self.expert_obs_ph: exp_obs, self.expert_acs_ph: exp_acs, self.labels_ph: labels, self.lprobs_ph: lprobs}
        reward = sess.run(self.Reward, feed_dict)
        return reward
