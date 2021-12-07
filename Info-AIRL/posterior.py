import tensorflow as tf
import numpy as np

from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common import tf_util as U
import gym
from baselines.acktr.utils import dense

class Posterior(object):
    def __init__(self, env, hidden_size, encode_num, scope="posterior"):
        self.ac_space = ac_space = env.action_space
        self.ob_space = ob_space = env.observation_space
        self.scope = scope
        self.observation_shape = ob_space.shape
        self.actions_shape = ac_space.shape
        self.encode_num = encode_num
        if isinstance(ac_space, gym.spaces.Box):
            self.num_actions = ac_space.shape[0]
        else:
            self.num_actions = ac_space.n

        self.input_shape = tuple([o+a for o, a in zip(self.observation_shape, self.actions_shape)])
        #print(env.action_space)


        self.hidden_size = hidden_size
        self.build_ph()
        # Build grpah

        posterior_logits = self.build_graph(self.generator_obs_ph, self.generator_acs_ph, encode_num, reuse=False)

        post_loss = -tf.reduce_mean(tf.reduce_sum(tf.log(posterior_logits)*self.encode_ph, axis=1))

        # Build regnum loss

        self.losses = [post_loss]
        self.loss_name = ["post_loss"]
        self.total_loss = post_loss
        # Build Reward for policy

        self.Reward = tf.reduce_sum(tf.log(posterior_logits)*self.encode_ph, axis=1)

        var_list = self.get_trainable_variables()

        self.lossandgrad = U.function([self.generator_obs_ph, self.generator_acs_ph, self.encode_ph],
                                      self.losses + [U.flatgrad(self.total_loss, var_list)])

    def build_ph(self):
        self.generator_obs_ph = tf.placeholder(tf.float32, (None, ) + self.observation_shape, name="observations_ph")
        self.generator_acs_ph = tf.placeholder(tf.float32, (None, ) + self.actions_shape, name="actions_ph")

        self.encode_ph = tf.placeholder(tf.float32, (None, self.encode_num), name='encode_ph')

    def build_graph(self, obs_ph, acs_ph, encode_num, reuse=False):
        with tf.variable_scope(self.scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("obfilter"):
                self.obs_rms = RunningMeanStd(shape=self.observation_shape)
            obs = (obs_ph - self.obs_rms.mean) / self.obs_rms.std
            if isinstance(self.ac_space, gym.spaces.Discrete):
                #print(acs_ph)
                acs_ph = tf.one_hot(tf.cast(acs_ph, tf.int32), self.num_actions, 1.0, 0.0)
                #print(acs_ph)
            _input = tf.concat([obs, acs_ph], axis=1)  # concatenate the two input -> form a transition

            '''p_h1 = tf.nn.relu(dense(_input, self.hidden_size, "posterior%i" % (1), weight_init=U.normc_initializer(1.0)))
            p_h2 = tf.nn.relu(dense(p_h1, self.hidden_size, "posterior%i" % (2), weight_init=U.normc_initializer(1.0)))
            logits = tf.nn.softmax(dense(p_h2, 3, "posterior%i" % (3), weight_init=U.normc_initializer(1.0)))'''
            p_h1 = tf.contrib.layers.fully_connected(_input, self.hidden_size, activation_fn=tf.nn.relu)
            p_h2 = tf.contrib.layers.fully_connected(p_h1, self.hidden_size, activation_fn=tf.nn.relu)
            logits = tf.contrib.layers.fully_connected(p_h2, encode_num, activation_fn=tf.nn.softmax)
            print("logits: ", logits)
        return logits

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_reward(self, obs, acs, encode):
        sess = tf.get_default_session()
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, 0)
        if len(acs.shape) == 1 and isinstance(self.ac_space, gym.spaces.Box):
            acs = np.expand_dims(acs, 0)

        feed_dict = {self.generator_obs_ph: obs, self.generator_acs_ph: acs, self.encode_ph: encode}
        reward = sess.run(self.Reward, feed_dict)
        return np.expand_dims(reward, axis=1)