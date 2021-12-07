'''
from baselines/ppo1/mlp_policy.py and add simple modification
(1) add reuse argument
(2) cache the `stochastic` placeholder
'''
import tensorflow as tf
import gym
import numpy as np
import baselines.common.tf_util as U
from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common.distributions import make_pdtype
from baselines.acktr.utils import dense


class MlpPolicy(object):


    def __init__(self, name, reuse=False, *args, **kwargs):
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name
            self.recurrent = False

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, encode_num, min_std = 1e-6):

        assert isinstance(ob_space, gym.spaces.Box)

        self.ac_space = ac_space

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))
        encode_ph = U.get_placeholder(name="encode_ph", dtype=tf.float32, shape=(None, encode_num))
        print("test: ", encode_num)

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)

        last_out = tf.concat([obz, encode_ph], axis=1)
        print("last_out_1: ", last_out)
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(dense(last_out, hid_size, "vffc%i" % (i+1), weight_init=U.normc_initializer(1.0)))

        self.vpred = dense(last_out, 1, "vffinal", weight_init=U.normc_initializer(1.0))[:, 0]

        last_out = obz
        #for i in range(num_hid_layers):
        last_out = tf.nn.tanh(dense(last_out, hid_size, "polfc%i" % (1), weight_init=U.normc_initializer(1.0)))
        last_out = dense(last_out, hid_size, "polfc%i" % (2), weight_init=U.normc_initializer(1.0))
        encode_out = dense(encode_ph, hid_size, "polfc%i" % (3), weight_init=U.normc_initializer(1.0))
        last_out = tf.nn.tanh(tf.add(last_out, encode_out))
        print("last_out_2: ", last_out)

        if isinstance(ac_space, gym.spaces.Box):
            self.mean = mean = dense(last_out, pdtype.param_shape()[0] // 2, "polfinal", U.normc_initializer(0.01))
            self.logstd = logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0] // 2],
                                     initializer=tf.zeros_initializer())
            pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)

            min_std_param = np.log(min_std)
            std_param_var = tf.maximum(logstd, min_std_param)

        else:
            pdparam = dense(last_out, pdtype.param_shape()[0], "polfinal", U.normc_initializer(0.01))

        self.pd = pdtype.pdfromflat(pdparam)

        if isinstance(ac_space, gym.spaces.Discrete) or isinstance(ac_space, gym.spaces.MultiDiscrete):
            #only the softmax result is used, std is not used in the calculation of P(a|s) for discrete conditions
            self.mean = mean = tf.nn.softmax(pdparam)
            std_param_var = tf.zeros_like(mean)

        # change for BC
        stochastic = U.get_placeholder(name="stochastic", dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self.ac = ac

        self._act = U.function([stochastic, ob, encode_ph], [ac, self.mean, std_param_var, self.vpred])


    def act(self, stochastic, ob, encode):

        ac1, mean1, logstd1, vpred1 = self._act(stochastic, ob[None], encode[None])
        return ac1[0], mean1[0], logstd1[0], vpred1[0]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)


