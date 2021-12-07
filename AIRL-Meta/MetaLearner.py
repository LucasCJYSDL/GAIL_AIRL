import tensorflow as tf
import baselines.common.tf_util as U
from baselines.common import zipsame, dataset
from baselines.common.mpi_adam import MpiAdam
from mpi4py import MPI
from train_util import compute_path_probs, allmean, add_vtarg_and_adv, timed
import numpy as np
from baselines import logger
from baselines.common.cg import cg
from variables import VariableState, average_vars, interpolate_vars

class TrpoSolver(object):

    def __init__(self, nworkers, rank, pi, oldpi, ob, ac, var_list, entcoeff):

        self.nworkers = nworkers
        self.rank = rank
        self.oldpi = oldpi
        self.pi = pi

        self.atarg = tf.placeholder(dtype=tf.float32, shape=[None])

        self.kloldnew = oldpi.pd.kl(pi.pd)
        self.meankl = tf.reduce_mean(self.kloldnew)
        self.ent = pi.pd.entropy()
        self.meanent = tf.reduce_mean(self.ent)
        self.meanent = tf.stop_gradient(self.meanent)
        self.entbonus = entcoeff * self.meanent

        self.ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))  # advantage * pnew / pold
        self.surrgain = tf.reduce_mean(self.ratio * self.atarg)
        self.optimgain = self.surrgain + self.entbonus
        self.losses = [self.optimgain, self.meankl, self.entbonus, self.surrgain, self.meanent]

        self.dist = self.meankl
        self.get_flat = U.GetFlat(var_list)
        self.set_from_flat = U.SetFromFlat(var_list)
        self.klgrads = tf.gradients(self.dist, var_list)
        self.flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
        self.shapes = [var.get_shape().as_list() for var in var_list]
        start = 0
        self.tangents = []
        for shape in self.shapes:
            sz = U.intprod(shape)
            self.tangents.append(tf.reshape(self.flat_tangent[start:start + sz], shape))
            start += sz
        self.gvp = tf.add_n(
            [tf.reduce_sum(g * tangent) for (g, tangent) in zipsame(self.klgrads, self.tangents)])  # pylint: disable=E1111
        self.fvp = U.flatgrad(self.gvp, var_list)

        self.compute_losses = U.function([ob, ac, self.atarg], self.losses)
        self.compute_lossandgrad = U.function([ob, ac, self.atarg], self.losses + [U.flatgrad(self.optimgain, var_list)])
        self.compute_fvp = U.function([self.flat_tangent, ob, ac, self.atarg], self.fvp)

    def step_P(self, args, assign_old_eq_new, cg_damping, max_kl, cg_iters):
        fvpargs = [arr[::5] for arr in args]

        def fisher_vector_product(p):
            return allmean(self.compute_fvp(p, *fvpargs), self.nworkers) + cg_damping * p

        with timed("computegrad", self.rank):
            *lossbefore, g = self.compute_lossandgrad(*args)
            lossbefore = allmean(np.array(lossbefore), self.nworkers)
            g = allmean(g, self.nworkers)

        # print("before assign oldpi: ", U.get_session().run(self.oldpi.get_trainable_variables()[0]))
        # print("before assign pi: ", U.get_session().run(self.pi.get_trainable_variables()[0]))
        # print("before assign: ", id(self.pi), "   ", id(self.oldpi))
        assign_old_eq_new()
        # print("after assign oldpi: ", U.get_session().run(self.oldpi.get_trainable_variables()[0]))

        if np.allclose(g, 0):
            logger.log("Got zero gradient. not updating")
        else:
            with timed("cg", self.rank):
                stepdir = cg(fisher_vector_product, g, cg_iters=cg_iters, verbose= self.rank == 0)
            assert np.isfinite(stepdir).all()
            shs = .5 * stepdir.dot(fisher_vector_product(stepdir))
            lm = np.sqrt(shs / max_kl)
            # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
            fullstep = stepdir / lm
            expectedimprove = g.dot(fullstep)
            surrbefore = lossbefore[0]
            stepsize = 1.0
            thbefore = self.get_flat()
            for _ in range(10):
                thnew = thbefore + fullstep * stepsize
                self.set_from_flat(thnew)
                meanlosses = surr, kl, *_ = allmean(np.array(self.compute_losses(*args)), self.nworkers)
                improve = surr - surrbefore
                logger.log("Expected: %.3f Actual: %.3f" % (expectedimprove, improve))
                if not np.isfinite(meanlosses).all():
                    logger.log("Got non-finite value of losses -- bad!")
                elif kl > max_kl * 1.5:
                    logger.log("violated KL constraint. shrinking step.")
                elif improve < 0:
                    logger.log("surrogate didn't improve. shrinking step.")
                else:
                    logger.log("Stepsize OK!")
                    break
                stepsize *= .5
            else:
                logger.log("couldn't compute a good step")
                self.set_from_flat(thbefore)


        return meanlosses


    def initialize(self):
        th_init = self.get_flat()
        MPI.COMM_WORLD.Bcast(th_init, root=0)
        self.set_from_flat(th_init)
        if self.rank == 0:
            print("Init param sum", th_init.sum(), flush=True)


class MetaLearner(object):

    def __init__(self, env, nworkers, rank, pi, oldpi, discriminator, entcoeff):

        # self.sess = tf.Session()
        self.env = env
        self.nworkers = nworkers
        self.rank = rank

        self.discriminator = discriminator
        self.pi = pi
        self.oldpi =oldpi

        self.assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
                                                    for (oldv, newv) in zipsame(self.oldpi.get_variables(), self.pi.get_variables())])

        self.ob = U.get_placeholder_cached(name="ob")
        self.ac = self.pi.pdtype.sample_placeholder([None])
        self.ret = tf.placeholder(dtype=tf.float32, shape=[None])
        self.vferr = tf.reduce_mean(tf.square(self.pi.vpred - self.ret))
        self.all_var_list = self.pi.get_trainable_variables()
        self.var_list = [v for v in self.all_var_list if v.name.startswith("pi/pol") or v.name.startswith("pi/logstd")]
        self.vf_var_list = [v for v in self.all_var_list if v.name.startswith("pi/vff")]
        self.dis_var_list = self.discriminator.get_trainable_variables()

        self.pi_para_model = VariableState(self.all_var_list)
        self.dis_para_model = VariableState(self.dis_var_list)

        self.d_adam = MpiAdam(self.dis_var_list)
        self.vfadam = MpiAdam(self.vf_var_list)

        self.compute_vflossandgrad = U.function([self.ob, self.ret], U.flatgrad(self.vferr, self.vf_var_list))

        self.trpo = TrpoSolver(self.nworkers, self.rank, self.pi, self.oldpi, self.ob, self.ac, self.var_list, entcoeff)


    def initialize(self):
        U.initialize()
        print("mpi: ", id(self.pi))
        print("moldpi", id(self.oldpi))
        print("mdisc: ", id(self.discriminator))
        self.trpo.initialize()
        self.d_adam.sync()
        self.vfadam.sync()

    def load_model(self, path):
        U.load_state(path)

    def save_para(self):
        self.pi_paras = self.pi_para_model.export_variables()
        self.dis_paras = self.dis_para_model.export_variables()
        return self.pi_paras, self.dis_paras

    def restore_para(self, pi_before_paras, dis_before_paras):
        self.pi_para_model.import_variables(pi_before_paras)
        self.dis_para_model.import_variables(dis_before_paras)

    def meta_update(self, meta_stepsize, pi_after_paras, dis_after_paras, pi_before_paras, dis_before_paras):
        pi_after_paras = average_vars(pi_after_paras)
        dis_after_paras = average_vars(dis_after_paras)
        # print("9: ", pi_after_paras[0])
        # print("10: ", dis_after_paras[0])
        self.pi_para_model.import_variables(interpolate_vars(pi_before_paras, pi_after_paras, meta_stepsize))
        self.dis_para_model.import_variables(interpolate_vars(dis_before_paras, dis_after_paras, meta_stepsize))
        p, d = self.save_para()
        # print("11: ", p[0])
        # print("12: ", d[0])
        # print("13: ", meta_stepsize)

    def update_D(self, ob, ac, lprobs, batch_size, expert_dataset, labels, d_stepsize):

        ac_space = self.env.action_space
        d_losses = []
        for ob_batch, ac_batch, lprobs_batch in dataset.iterbatches((ob, ac, lprobs), shuffle=True,
                                                                    include_final_partial_batch=False,
                                                                    batch_size=batch_size):
            ob_expert, ac_expert, mean_expert, std_expert = expert_dataset.get_next_batch(len(ob_batch))
            lprobs_expert = compute_path_probs(ac_expert, mean_expert, std_expert, ac_space, is_expert=True)
            # update running mean/std for discriminator
            if hasattr(self.discriminator, "obs_rms"): self.discriminator.obs_rms.update(np.concatenate((ob_batch, ob_expert), 0))
            lprob_batch = np.concatenate([lprobs_batch, lprobs_expert], axis=0)

            *newlosses, g = self.discriminator.lossandgrad(ob_batch, ac_batch, ob_expert, ac_expert, labels, lprob_batch)
            self.d_adam.update(allmean(g, self.nworkers), d_stepsize)
            d_losses.append(newlosses)

        return d_losses

    def update_P(self, ob, ac, atarg, cg_damping, max_kl, cg_iters):

        if hasattr(self.pi, "ob_rms"):
            self.pi.ob_rms.update(ob)

        args = ob, ac, atarg

        return self.trpo.step_P(args, self.assign_old_eq_new, cg_damping, max_kl, cg_iters)

    def update_V(self, vf_iters, ob, tdlamret, vf_stepsize):

        with timed("vf", self.rank):
            for _ in range(vf_iters):
                for (mbob, mbret) in dataset.iterbatches((ob, tdlamret), shuffle=True,
                                                                include_final_partial_batch=False, batch_size=128):
                    if hasattr(self.pi, "ob_rms"):
                        self.pi.ob_rms.update(mbob)  # update running mean/std for policy
                    g = allmean(self.compute_vflossandgrad(mbob, mbret), self.nworkers)
                    self.vfadam.update(g, vf_stepsize)

    def getRewardAndAdvantage(self, segs, coe_d, coe_t, use_true_reward, use_sparse_reward, gamma, lam):

        Disc_rewards = []
        Total_rewards = []

        for seg in segs:
            new_batch_size = seg["ob"].shape[0]
            lprobs_rew = seg["lprobs"]
            Disc_reward = self.discriminator.get_reward(seg["ob"], seg["ac"], new_batch_size, lprobs_rew)

            rewards = coe_d * Disc_reward

            if use_true_reward:
                true_rew = np.expand_dims(seg["true_reward"], axis=1)
                rewards += coe_t * true_rew  ##reward_2

            elif use_sparse_reward:
                print('rewards: ', rewards)
                sparse_rew = np.expand_dims(seg["sparse_reward"], axis=1)
                print("sparse: ", sparse_rew)
                rewards += coe_t * sparse_rew  ##reward_2

            seg["rew"] = rewards


            Disc_rewards.append(np.sum(Disc_reward))
            Total_rewards.append(np.sum(rewards))
            add_vtarg_and_adv(seg, gamma, lam)


        return np.array(Disc_rewards).mean(), np.array(Total_rewards).mean()