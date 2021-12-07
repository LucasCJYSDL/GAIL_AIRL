'''
from baselines/ppo1/mlp_policy.py and add simple modification
(1) add reuse argument
(2) cache the `stochastic` placeholder
'''
import tensorflow as tf
import gym

import common.tf_util as U
from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common.distributions import make_pdtype
from baselines.acktr.utils import dense
from common import utils
from common.input import observation_placeholder

import numpy as np


class MlpPolicy(object):
    recurrent = False

    def __init__(self, name, reuse=False, *args, **kwargs):
        with tf.variable_scope(name):
            if reuse:##
                tf.get_variable_scope().reuse_variables()##
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)###1
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))#batch_size*shape的placeholder

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)#剔除一些不好的观测###2
        ##
        obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
        last_out = obz
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(dense(last_out, hid_size, "vffc%i" % (i+1), weight_init=U.normc_initializer(1.0)))
        self.vpred = dense(last_out, 1, "vffinal", weight_init=U.normc_initializer(1.0))[:, 0]

        last_out = obz
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(dense(last_out, hid_size, "polfc%i" % (i+1), weight_init=U.normc_initializer(1.0)))

        if isinstance(ac_space, gym.spaces.Box):###3
            mean = dense(last_out, pdtype.param_shape()[0]//2, "polfinal", U.normc_initializer(0.01))
            logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
            pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
        else:
            pdparam = dense(last_out, pdtype.param_shape()[0], "polfinal", U.normc_initializer(0.01))

        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        # change for BC
        stochastic = U.get_placeholder(name="stochastic", dtype=tf.bool, shape=())##???
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self.ac = ac##
        self._act = U.function([stochastic, ob], [ac, self.vpred])


    def act(self, stochastic, ob):
        ac1, vpred1 = self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)


class MlpGruPolicy(object):

    def __init__(self, name, ob_space, ac_space, nbatch, num_hid_layers = 2, hid_size = 100, ngru=128, nbatch_vf=8, nbatch_pol=16, reuse=False):

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            if reuse:
                tf.get_variable_scope().reuse_variables()  ##

            self.recurrent = True
            self.ob = ob = observation_placeholder(ob_space, batch_size=nbatch)
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)  # 剔除一些不好的观测###2

            obz = tf.clip_by_value((tf.cast(ob, tf.float32) - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
            if nbatch%nbatch_vf==0:
                v_o = tf.split(axis=0, num_or_size_splits=nbatch_vf, value=obz)

            else:
                v_o = [obz]

            vpred = []
            for v in v_o:
                last_out = v
                for i in range(num_hid_layers):
                    last_out = tf.nn.tanh(dense(last_out, hid_size, "vffc%i" % (i + 1), weight_init=U.normc_initializer(1.0)))
                vpred.append(dense(last_out, 1, "vffinal", weight_init=U.normc_initializer(1.0))[:, 0])
            temp_vpred = []
            for i in range(len(vpred)):
                for j in range(vpred[i].shape[0]):
                    temp_vpred.append(vpred[i][j])
            self.vpred = tf.convert_to_tensor(temp_vpred)

            #print(self.vpred)
            self.M = M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
            self.S = S = tf.placeholder(tf.float32, [1, ngru]) #states

            if nbatch%nbatch_pol==0:
                p_o = tf.split(axis=0, num_or_size_splits=nbatch_pol, value=obz)
                p_M = tf.split(axis=0, num_or_size_splits=nbatch_pol, value=M)
                num_batch = nbatch//nbatch_pol

            else:
                p_o = [obz]
                p_M = [M]
                num_batch = nbatch

            pdparam = []
            for i in range(len(p_o)):
                p = p_o[i]
                m = p_M[i]
                h1 = tf.layers.flatten(p)
                xs = utils.batch_to_seq(h1, 1, num_batch)
                ms = utils.batch_to_seq(m, 1, num_batch)
                h2, snew ,self. w, self.ww, self.b= utils.gru(xs, ms, S, 'polgru', nh=ngru)
                h2 = utils.seq_to_batch(h2)
                self.h = h = tf.layers.flatten(h2)#?????
                self.pdtype = pdtype = make_pdtype(ac_space)
                mean = dense(h, pdtype.param_shape()[0] // 2, "polfinal", U.normc_initializer(0.01))
                logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0] // 2],
                                        initializer=tf.zeros_initializer())
                pdparam.append(tf.concat([mean, mean * 0.0 + logstd], axis=1))

            temp_param = []
            for i in range(len(pdparam)):
                for j in range(pdparam[i].shape[0]):
                    temp_param.append(pdparam[i][j])

            self.pdparam = tf.convert_to_tensor(temp_param)

            #print(self.pdparam)
            self.pd = pdtype.pdfromflat(self.pdparam)
            self.action = self.pd.sample()
            self.initial_state = np.zeros((1, ngru), dtype=np.float32)
            self.logp = self.pd.logp(self.action)
            self._act = U.function([self.ob, self.S, self.M], [self.action, self.vpred, snew, self.logp, self.pdparam])
            self.scope = tf.get_variable_scope().name
            self._value = U.function([self.ob, self.S, self.M], self.vpred)

    def step(self, ob, state, mask):
        ac1, vf1, S1, logp, pdparam= self._act(ob, state, mask)
        return ac1, vf1, S1, logp, pdparam

    def value(self, ob, state, mask):
        vf1 = self._value(ob, state, mask)
        return vf1

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

class GruPolicy(object):

    def __init__(self, name, ob_space, ac_space, nbatch, ngru=128, reuse=False):

        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()  ##

            self.recurrent = True
            self.ob = ob = observation_placeholder(ob_space, batch_size=nbatch)
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)  # 剔除一些不好的观测###2
            obz = tf.clip_by_value((tf.cast(ob, tf.float32) - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)

            self.M = M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
            self.S = S = tf.placeholder(tf.float32, [1, ngru]) #states

            h1 = tf.layers.flatten(obz)
            xs = utils.batch_to_seq(h1, 1, nbatch)
            ms = utils.batch_to_seq(M, 1, nbatch)
            h2, snew ,self. w, self.ww, self.b= utils.gru(xs, ms, S, 'gru', nh=ngru)
            h2 = utils.seq_to_batch(h2)
            self.h = h = tf.layers.flatten(h2)
            vf = dense(h, 1, "v", weight_init=U.normc_initializer(1.0))
            self.vpred = vf[:, 0]

            self.pdtype = make_pdtype(ac_space)
            self.pd, self.pi = self.pdtype.pdfromlatent(h, init_scale=0.01)
            self.action = self.pd.sample()##可能改改

            self.initial_state = np.zeros((1, ngru), dtype=np.float32)
            self.logp = self.pd.logp(self.action)
            self._act = U.function([ob, S, M], [self.action, self.vpred, snew, self.logp])
            self.scope = tf.get_variable_scope().name
            self._value = U.function([self.ob, self.S, self.M], self.vpred)

    def step(self, ob, state, mask):
        ac1, vf1, S1, logp = self._act(ob, state, mask)
        return ac1, vf1, S1, logp

    def value(self, ob, state, mask):
        vf1 = self._value(ob, state, mask)
        return vf1

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

class LstmPolicy(object):

    def __init__(self, name, ob_space, ac_space, nbatch, nlstm=128, reuse=False):

        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()  ##

            self.recurrent = True
            self.ob = ob = observation_placeholder(ob_space, batch_size=nbatch)
            self.M = M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
            self.S = S = tf.placeholder(tf.float32, [1, 2*nlstm]) #states


            h1 = tf.layers.flatten(tf.cast(ob, tf.float32))
            xs = utils.batch_to_seq(h1, 1, nbatch)
            ms = utils.batch_to_seq(M, 1, nbatch)
            h2, snew = utils.lstm(xs, ms, S, 'lstm', nh=nlstm)
            h2 = utils.seq_to_batch(h2)
            h = tf.layers.flatten(h2)
            vf = utils.fc(h, 'v', 1)
            self.vpred = vf[:, 0]

            self.pdtype = make_pdtype(ac_space)
            self.pd, self.pi = self.pdtype.pdfromlatent(h, init_scale=0.01)
            self.action = self.pd.sample()##可能改改
            self.logp = self.pd.logp(self.action)
            self.initial_state = np.zeros((1, 2*nlstm), dtype=np.float32)
            self._act = U.function([self.ob, self.S, self.M], [self.action, self.vpred, snew, self.logp])
            self._value = U.function([self.ob, self.S, self.M], self.vpred)
            self.scope = tf.get_variable_scope().name

    def step(self, ob, state, mask):
        ac1, vf1, S1, logp = self._act(ob, state, mask)
        return ac1, vf1, S1, logp

    def value(self, ob, state, mask):
        vf1 = self._value(ob, state, mask)
        return vf1

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)