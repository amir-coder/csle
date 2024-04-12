import copy
import sys
from mcs_utils.mcs_fun import MCSUtils
from mcs_utils.gls_utils import GLSUtils
from mcs_utils.ls_utils import LSUtils

from numpy.typing import NDArray
from typing import Union, List, Optional, Any, Dict
import gymnasium as gym
import numpy as np
import gym_csle_stopping_game.constants.constants as env_constants
from csle_common.dao.emulation_config.emulation_env_config import EmulationEnvConfig
from csle_common.dao.simulation_config.simulation_env_config import SimulationEnvConfig
from csle_common.dao.training.experiment_config import ExperimentConfig
# from csle_common.dao.training.experiment_execution import ExperimentExecution
# from csle_common.dao.training.experiment_result import ExperimentResult
from csle_common.dao.training.agent_type import AgentType
from csle_common.dao.training.player_type import PlayerType
# from csle_common.util.experiment_util import ExperimentUtil
from csle_common.logging.log import Logger
from csle_common.dao.training.multi_threshold_stopping_policy import MultiThresholdStoppingPolicy
from csle_common.dao.training.linear_threshold_stopping_policy import LinearThresholdStoppingPolicy
# from csle_common.metastore.metastore_facade import MetastoreFacade
from csle_common.dao.jobs.training_job_config import TrainingJobConfig
# from csle_common.util.general_util import GeneralUtil
from csle_common.dao.simulation_config.base_env import BaseEnv
from csle_common.dao.training.policy_type import PolicyType
from csle_agents.agents.base.base_agent import BaseAgent
# from csle_agents.common.objective_type import ObjectiveType
import csle_agents.constants.constants as agents_constants


class MCSAgent(BaseAgent):
    def __init__(self, simulation_env_config: SimulationEnvConfig,
                 emulation_env_config: Union[None, EmulationEnvConfig],
                 experiment_config: ExperimentConfig,
                 env: Optional[BaseEnv] = None, training_job: Optional[TrainingJobConfig] = None,
                 save_to_metastore: bool = True):
        """
        Initializes the MCS Agent

        :param simulation_env_config: the simulation env config
        :param emulation_env_config: the emulation env config
        :param experiment_config: the experiment config
        :param env: (optional) the gym environment to use for simulation
        :param training_job: (optional) a training job configuration
        :param save_to_metastore: boolean flag that can be set to avoid saving results and progress to the metastore
        """
        super().__init__(simulation_env_config=simulation_env_config, emulation_env_config=emulation_env_config,
                         experiment_config=experiment_config)
        assert experiment_config.agent_type == AgentType.MCS
        self.env = env
        self.training_job = training_job
        self.save_to_metastore = save_to_metastore

    def eval_theta(self, policy: Union[MultiThresholdStoppingPolicy, LinearThresholdStoppingPolicy],
                   max_steps: int = 200) -> Dict[str, Union[float, int]]:
        """
        Evaluates a given threshold policy by running monte-carlo simulations

        :param policy: the policy to evaluate
        :return: the average metrics of the evaluation
        """
        if self.env is None:
            raise ValueError("Need to specify an environment to run policy evaluation")
        eval_batch_size = self.experiment_config.hparams[agents_constants.COMMON.EVAL_BATCH_SIZE].value
        metrics: Dict[str, Any] = {}
        for j in range(eval_batch_size):
            done = False
            o, _ = self.env.reset()
            l = int(o[0])
            b1 = o[1]
            t = 1
            r = 0
            a = 0
            info: Dict[str, Any] = {}
            while not done and t <= max_steps:
                Logger.__call__().get_logger().debug(f"t:{t}, a: {a}, b1:{b1}, r:{r}, l:{l}, info:{info}")
                if self.experiment_config.player_type == PlayerType.ATTACKER:
                    policy.opponent_strategy = self.env.static_defender_strategy
                    a = policy.action(o=o)
                else:
                    a = policy.action(o=o)
                o, r, done, _, info = self.env.step(a)
                print(info)
                l = int(o[0])
                b1 = o[1]
                t += 1
            metrics = MCSAgent.update_metrics(metrics=metrics, info=info)
        avg_metrics = MCSAgent.compute_avg_metrics(metrics=metrics)
        return avg_metrics

    @staticmethod
    def update_metrics(metrics: Dict[str, List[Union[float, int]]], info: Dict[str, Union[float, int]]) \
            -> Dict[str, List[Union[float, int]]]:
        """
        Update a dict with aggregated metrics using new information from the environment

        :param metrics: the dict with the aggregated metrics
        :param info: the new information
        :return: the updated dict
        """
        for k, v in info.items():
            if k in metrics:
                metrics[k].append(round(v, 3))
            else:
                metrics[k] = [v]
        return metrics

    @staticmethod
    def compute_avg_metrics(metrics: Dict[str, List[Union[float, int]]]) -> Dict[str, Union[float, int]]:
        """
        Computes the average metrics of a dict with aggregated metrics

        :param metrics: the dict with the aggregated metrics
        :return: the average metrics
        """
        avg_metrics = {}
        for k, v in metrics.items():
            avg = round(sum(v) / len(v), 2)
            avg_metrics[k] = avg
        return avg_metrics

    def hparam_names(self) -> List[str]:
        """
        :return: a list with the hyperparameter names
        """
        return [agents_constants.MCS.STEP, agents_constants.MCS.STEP1,
                agents_constants.MCS.U, agents_constants.MCS.V,
                agents_constants.MCS.LOCAL, agents_constants.MCS.STOPPING_ACTIONS,
                agents_constants.MCS.GAMMA, agents_constants.MCS.EPSILON,
                agents_constants.COMMON.CONFIDENCE_INTERVAL,
                agents_constants.COMMON.RUNNING_AVERAGE]

    def train(self):
        """
        Initiating the parameters of performing the MCS algorithm, using external functions
        """
        u = self.experiment_config.hparams[agents_constants.MCS.U].value
        v = self.experiment_config.hparams[agents_constants.MCS.V].value
        iinit = self.experiment_config.hparams[agents_constants.MCS.IINIT].value
        local = self.experiment_config.hparams[agents_constants.MCS.LOCAL].value
        eps = self.experiment_config.hparams[agents_constants.MCS.EPSILON].value
        gamma = self.experiment_config.hparams[agents_constants.MCS.GAMMA].value
        # prt = self.experiment_config.hparams[agents_constants.MCS.PRT].value
        # m = self.experiment_config.hparams[agents_constants.MCS.M].value
        stopping_actions = self.experiment_config.hparams[agents_constants.MCS.STOPPING_ACTIONS].value
        n = len(u)
        smax = 5 * n + 10
        nf = 50 * pow(n, 2)
        stop = [3 * n]
        hess = np.ones((n, n))
        stop.append(float("-inf"))

        config = self.simulation_env_config.simulation_env_input_config
        if self.env is None:
            self.env = gym.make(self.simulation_env_config.gym_env_name, config=config)

        xbest, fbest, xmin, fmi, ncall, ncloc, flag = self.MCS(u, v, smax, nf, stop, iinit,
                                                               local, gamma, hess, stopping_actions, eps, n)
        print('The MCS Algorithms Results:') # TODO: make log-statements
        print('fbest', fbest)
        print('xbest', xbest)
        print('\n')

    def get_policy(self, theta: List[float], L: int) -> Union[MultiThresholdStoppingPolicy,
                                                              LinearThresholdStoppingPolicy]:
        """
        Gets the policy of a given parameter vector

        :param theta: the parameter vector
        :param L: the number of parameters
        :return: the policy
        """
        if self.experiment_config.hparams[agents_constants.SIMULATED_ANNEALING.POLICY_TYPE].value \
                == PolicyType.MULTI_THRESHOLD.value:
            policy = MultiThresholdStoppingPolicy(
                theta=list(theta), simulation_name=self.simulation_env_config.name,
                states=self.simulation_env_config.state_space_config.states,
                player_type=self.experiment_config.player_type, L=L,
                actions=self.simulation_env_config.joint_action_space_config.action_spaces[
                    self.experiment_config.player_idx].actions, experiment_config=self.experiment_config, avg_R=-1,
                agent_type=AgentType.SIMULATED_ANNEALING)
        else:
            policy = LinearThresholdStoppingPolicy(
                theta=list(theta), simulation_name=self.simulation_env_config.name,
                states=self.simulation_env_config.state_space_config.states,
                player_type=self.experiment_config.player_type, L=L,
                actions=self.simulation_env_config.joint_action_space_config.action_spaces[
                    self.experiment_config.player_idx].actions, experiment_config=self.experiment_config, avg_R=-1,
                agent_type=AgentType.SIMULATED_ANNEALING)
        return policy

    def init_list(self, theta0: NDArray[np.int32], l: NDArray[np.int32], L: NDArray[np.int32], stopping_actions: int, n: int):
        '''
        computes the function values corresponding to the initialization list
        and the pointer istar to the final best point x^* of the init. list
        '''
        ncall = 0
  
        theta = np.zeros(n)
        for i in range(n):
            theta[i] = theta0[i, l[i]]

        policy = self.get_policy(theta, L=stopping_actions)
        avg_metrics = self.eval_theta(policy=policy,
                                      max_steps=self.experiment_config.hparams[
                                          agents_constants.COMMON.MAX_ENV_STEPS].value)
        J1 = round(avg_metrics[env_constants.ENV_METRICS.RETURN], 3)
        print("J1 = ", J1)
        ncall += 1
        
        J0 = np.zeros((L[0] + 1, n))
        J0[l[0], 0] = J1

        istar = np.zeros(n).astype(int)

        for i in range(n):
            istar[i] = l[i]
            for j in range(L[i] + 1):
                if j == l[i]:
                    if i != 0:
                        J0[j, i] = J0[istar[i - 1], i - 1]
                else:
                    theta[i] = theta0[i, j]
                    policy = self.get_policy(theta, L=stopping_actions)
                    avg_metrics = self.eval_theta(policy=policy,
                                                  max_steps=self.experiment_config.hparams[
                                                      agents_constants.COMMON.MAX_ENV_STEPS].value)
                    J0[j, i] = round(avg_metrics[env_constants.ENV_METRICS.RETURN], 3)

                    ncall = ncall + 1

                    if J0[j, i] < J1:
                        J1 = J0[j, i]
                        istar[i] = j

            theta[i] = theta0[i, istar[i]]
        return J0, istar, ncall

    def MCS(self, u: List[int], v: List[int], smax: int, nf: int, stop: List[int], iinit: int,
            local: int, gamma: float, hess: NDArray[np.float64], stopping_actions: int,
            eps: float, n: int, prt: int=1):

        if MCSUtils().check_box_bound(u, v):
            sys.exit("Error MCS main: out of bound")
        n = len(u)
        ncall = 0
        ncloc = 0

        l = np.multiply(1, np.ones(n)).astype(int)
        L = np.multiply(2, np.ones(n)).astype(int)
        theta0 = MCSUtils().get_theta0(iinit, u, v, n)
        if iinit != 3:
            f0, istar, ncall1 = self.init_list(theta0, l, L, stopping_actions, n)
            ncall = ncall + ncall1
        theta = np.zeros(n)
        for i in range(n):
            theta[i] = theta0[i, l[i]]
        v1 = np.zeros(n)
        for i in range(n):
            if abs(theta[i] - u[i]) > abs(theta[i] - v[i]):
                v1[i] = u[i]
            else:
                v1[i] = v[i]

        step = self.experiment_config.hparams[agents_constants.MCS.STEP].value
        step1 = self.experiment_config.hparams[agents_constants.MCS.STEP1].value
        dim = step1

        isplit = np.zeros(step1).astype(int)
        level = np.zeros(step1).astype(int)
        ipar = np.zeros(step1).astype(int)
        ichild = np.zeros(step1).astype(int)
        nogain = np.zeros(step1).astype(int)

        f = np.zeros((2, step1))
        z = np.zeros((2, step1))

        record = np.zeros(smax)
        nboxes = 0
        nbasket = -1
        nbasket0 = -1
        nsweepbest = 0
        nsweep = 0
        m = n
        record[0] = 1
        nloc = 0
        xloc = []
        flag = 1
        ipar, level, ichild, f, isplit, p, xbest, fbest, nboxes = MCSUtils().initbox(
            theta0, f0, l, L, istar, u, v, isplit, level, ipar, ichild, f, nboxes, prt
        )
        f0min = fbest

        if stop[0] > 0 and stop[0] < 1:
            flag = MCSUtils().chrelerr(fbest, stop)
        elif stop[0] == 0:
            flag = MCSUtils().chvtr(fbest, stop[1])
        if not flag:
            print("global minumum as been found :", flag)

        s, record = MCSUtils().strtsw(smax, level, f[0, :], nboxes, record)
        nsweep = nsweep + 1
        xmin = []
        fmi = []
        while s < smax and ncall + 1 <= nf:
            par = record[s]
            n0, x, y, x1, x2, f1, f2 = MCSUtils().vertex(
                par, n, u, v, v1, theta0, f0, ipar, isplit, ichild, z, f, l, L
            )

            if s > 2 * n * (min(n0) + 1):
                isplit[par], z[1, par] = MCSUtils().splrnk(n, n0, p, x, y)
                splt = 1
            else:
                if nogain[par]:
                    splt = 0
                else:
                    e, isplit[par], z[1, par] = MCSUtils().exgain(
                        n, n0, l, L, x, y, x1, x2, f[0, par], f0, f1, f2
                    )
                    fexp = f[0, par] + min(e)
                    if fexp < fbest:
                        splt = 1
                    else:
                        splt = 0
                        nogain[par] = (1)

            if splt == 1:
                i = isplit[par]
                level[par] = 0
                if z[1, par] == np.Inf:
                    m = m + 1
                    z[1, par] = m # TODO : make fewer rows
                    (
                        xbest,
                        fbest,
                        f01,
                        xmin,
                        fmi,
                        ipar,
                        level,
                        ichild,
                        f,
                        flag,
                        ncall1,
                        record,
                        nboxes,
                        nbasket,
                        nsweepbest,
                        nsweep,
                    ) = self.splinit(
                        i,
                        s,
                        smax,
                        par,
                        theta0,
                        n0,
                        u,
                        v,
                        x,
                        y,
                        x1,
                        x2,
                        L,
                        l,
                        xmin,
                        fmi,
                        ipar,
                        level,
                        ichild,
                        f,
                        xbest,
                        fbest,
                        stop,
                        prt,
                        record,
                        nboxes,
                        nbasket,
                        nsweepbest,
                        nsweep,
                        stopping_actions
                    )
                    f01 = f01.reshape(len(f01), 1)
                    f0 = np.concatenate((f0, f01), axis=1)
                    ncall = ncall + ncall1
                else:
                    z[0, par] = x[i]
                    (
                        xbest,
                        fbest,
                        xmin,
                        fmi,
                        ipar,
                        level,
                        ichild,
                        f,
                        flag,
                        ncall1,
                        record,
                        nboxes,
                        nbasket,
                        nsweepbest,
                        nsweep,
                    ) = self.split(
                        i,
                        s,
                        smax,
                        par,
                        n0,
                        u,
                        v,
                        x,
                        y,
                        x1,
                        x2,
                        z[:, par],
                        xmin,
                        fmi,
                        ipar,
                        level,
                        ichild,
                        f,
                        xbest,
                        fbest,
                        stop,
                        prt,
                        record,
                        nboxes,
                        nbasket,
                        nsweepbest,
                        nsweep,
                        stopping_actions
                    )
                    ncall = ncall + ncall1

                if nboxes > dim:
                    isplit = np.concatenate((isplit, np.zeros(step)))
                    level = np.concatenate((level, np.zeros(step)))
                    ipar = np.concatenate((ipar, np.zeros(step)))
                    ichild = np.concatenate((ichild, np.zeros(step)))
                    nogain = np.concatenate((nogain, np.zeros(step)))
                    J = np.concatenate((J, np.ones((2, step))), axis=1)
                    z = np.concatenate((z, np.ones((2, step))), axis=1)
                    dim = nboxes + step
                if not flag:
                    break
            else:
                if s + 1 < smax:
                    level[par] = s + 1
                    record = MCSUtils().updtrec(par, s + 1, f[0, :], record)
                else:
                    level[par] = 0
                    nbasket = nbasket + 1
                    if len(xmin) == nbasket:
                        xmin.append(copy.deepcopy(x))
                        fmi.append(f[0, par])
                    else:
                        xmin[nbasket] = copy.deepcopy(x)
                        fmi[nbasket] = f[0, par]
            s = s + 1
            while s < smax:
                if record[s] == 0:
                    s = s + 1
                else:
                    break

            if s == smax:
                if local:
                    fmiTemp = fmi[nbasket0 + 1: nbasket + 1]
                    xminTemp = xmin[nbasket0 + 1: nbasket + 1]
                    j = np.argsort(fmiTemp)
                    fmiTemp = np.sort(fmiTemp)
                    xminTemp = [copy.deepcopy(xminTemp[jInd]) for jInd in j]
                    fmi[nbasket0 + 1: nbasket + 1] = fmiTemp
                    xmin[nbasket0 + 1: nbasket + 1] = xminTemp

                    for j in range(nbasket0 + 1, nbasket + 1):
                        x = copy.deepcopy(xmin[j])
                        f1 = copy.deepcopy(fmi[j])
                        loc = MCSUtils().chkloc(nloc, xloc, x)
                        if loc:
                            nloc, xloc = MCSUtils().addloc(nloc, xloc, x)
                            (
                                xbest,
                                fbest,
                                xmin,
                                fmi,
                                x,
                                f1,
                                loc,
                                flag,
                                ncall1,
                                nsweep,
                                nsweepbest,
                            ) = self.basket(
                                x,
                                f1,
                                xmin,
                                fmi,
                                xbest,
                                fbest,
                                stop,
                                nbasket0,
                                nsweep,
                                nsweepbest,
                                stopping_actions
                            )
                            ncall = ncall + ncall1
                            if not flag:
                                break
                            if loc:
                                xmin1, fmi1, nc, flag, nsweep, nsweepbest = self.lsearch(
                                    x,
                                    f1,
                                    f0min,
                                    u,
                                    v,
                                    nf - ncall,
                                    stop,
                                    local,
                                    gamma,
                                    hess,
                                    nsweep,
                                    nsweepbest,
                                    stopping_actions,
                                    eps
                                )
                                ncall = ncall + nc
                                ncloc = ncloc + nc
                                if fmi1 < fbest:
                                    xbest = copy.deepcopy(xmin1)
                                    fbest = copy.deepcopy(fmi1)
                                    nsweepbest = nsweep
                                    if not flag:
                                        nbasket0 = nbasket0 + 1
                                        nbasket = copy.deepcopy(nbasket0)
                                        if len(xmin) == nbasket:
                                            xmin.append(copy.deepcopy(xmin1))
                                            fmi.append(copy.deepcopy(fmi1))
                                        else:
                                            xmin[nbasket] = copy.deepcopy(xmin1)
                                            fmi[nbasket] = copy.deepcopy(fmi1)
                                        break

                                    if stop[0] > 0 and stop[0] < 1:
                                        flag = MCSUtils().chrelerr(fbest, stop)
                                    elif stop[0] == 0:
                                        flag = MCSUtils().chvtr(fbest, stop[1])
                                    if not flag:
                                        return xbest, fbest, xmin, fmi, ncall, ncloc, flag
                                (
                                    xbest,
                                    fbest,
                                    xmin,
                                    fmi,
                                    loc,
                                    flag,
                                    ncall1,
                                    nsweep,
                                    nsweepbest,
                                ) = self.basket1(
                                    np.array(xmin1),
                                    fmi1,
                                    xmin,
                                    fmi,
                                    xbest,
                                    fbest,
                                    stop,
                                    nbasket0,
                                    nsweep,
                                    nsweepbest,
                                    stopping_actions
                                )
                                ncall = ncall + ncall1
                                if not flag:
                                    break
                                if loc:
                                    nbasket0 = nbasket0 + 1
                                    if len(xmin) == nbasket0:
                                        xmin.append(copy.deepcopy(xmin1))
                                        fmi.append(copy.deepcopy(fmi1))
                                    else:
                                        xmin[nbasket0] = copy.deepcopy(xmin1)
                                        fmi[nbasket0] = copy.deepcopy(fmi1)
                                    fbest, xbest = MCSUtils().fbestloc(
                                        fmi, fbest, xmin, xbest, nbasket0, stop
                                    )
                                    if not flag:
                                        nbasket = nbasket0
                                        break
                    nbasket = copy.deepcopy(nbasket0)
                    if not flag:
                        break

                s, record = MCSUtils().strtsw(smax, level, f[0, :], nboxes, record)
                if prt:
                    minlevel = s
                    print("nsweep:", nsweep)
                    print("minlevel:", minlevel)
                    print("ncall:", ncall)
                    print("fbest:", fbest)
                    print("xbest: ", xbest)
                    print("\n")

                if stop[0] > 1:
                    if nsweep - nsweepbest >= stop[0]:
                        flag = 3
                        return xbest, fbest, xmin, fmi, ncall, ncloc, flag
                nsweep = nsweep + 1
        if ncall >= nf:
            flag = 2
        return xbest, fbest, xmin, fmi, ncall, ncloc, flag

    def splinit(self, i: int, s: int, smax: int, par: int, x0: NDArray[np.int32], n0: int, u: List[int], v: List[int],
                x: NDArray[np.float64], y: NDArray[np.float64], x1: NDArray[np.float64], x2: NDArray[np.float64], L: NDArray[np.int32], l: NDArray[np.int32],
                xmin: NDArray[np.float64], fmi: NDArray[np.float64], ipar: NDArray[np.int32], level: NDArray[np.int32], ichild: NDArray[np.int32],
                f: NDArray[np.float64], xbest: NDArray[np.float64], fbest: NDArray[np.float64], stop: List[np.int32], prt: int, record: NDArray[np.int32],
                nboxes: int, nbasket: int, nsweepbest: int, nsweep: int, stopping_actions: int):
        """
        Splitting box at specified level s according to an initialization list
        """

        ncall = 0
        f0 = np.zeros(max(L) + 1)
        flag = 1

        for j in range(L[i] + 1):
            if j != l[i]:
                x[i] = x0[i, j]

                policy = self.get_policy(x, L=stopping_actions)
                avg_metrics = self.eval_theta(policy=policy,
                                              max_steps=self.experiment_config.hparams[
                                                  agents_constants.COMMON.MAX_ENV_STEPS].value)
                f0[j] = round(avg_metrics[env_constants.ENV_METRICS.RETURN], 3)
                ncall = ncall + 1
                if f0[j] < fbest:
                    fbest = f0[j]
                    xbest = copy.deepcopy(x)
                    nsweepbest = copy.deepcopy(nsweep)
                    if stop[0] > 0 and stop[0] < 1:
                        flag = MCSUtils().chrelerr(fbest, stop)
                    elif stop[0] == 0:
                        flag = MCSUtils().chvtr(fbest, stop[2])
                    if not flag:
                        return xbest, fbest, f0, xmin, fmi, ipar, level, ichild, f,
                    flag, ncall, record, nboxes, nbasket, nsweepbest, nsweep
            else:
                f0[j] = f[0, par]
        if s + 1 < smax:
            nchild = 0
            if u[i] < x0[i, 0]:
                nchild = nchild + 1
                nboxes = nboxes + 1
                ipar[nboxes], level[nboxes], ichild[nboxes], f[0, nboxes] = MCSUtils().genbox(par, s + 1, -nchild,
                                                                                              f0[0])
                record = MCSUtils().updtrec(nboxes, level[nboxes], f[0, :], record)
            for j in range(L[i]):
                nchild = nchild + 1
                if f0[j] <= f0[j + 1] or s + 2 < smax:
                    nboxes = nboxes + 1
                    if f0[j] <= f0[j + 1]:
                        level0 = s + 1
                    else:
                        level0 = s + 2
                    ipar[nboxes], level[nboxes], ichild[nboxes], f[0, nboxes] = MCSUtils().genbox(par, level0,
                                                                                                  -nchild, f0[j])
                    record = MCSUtils().updtrec(nboxes, level[nboxes], f[0, :], record)
                else:
                    x[i] = x0[i, j]
                    nbasket = nbasket + 1
                    if (len(xmin) == nbasket):
                        xmin.append(copy.deepcopy(x))
                        fmi.append(f0[j])
                    else:
                        xmin[nbasket] = copy.deepcopy(x)
                        fmi[nbasket] = f0[j]
                nchild = nchild + 1
                if f0[j + 1] < f0[j] or s + 2 < smax:
                    nboxes = nboxes + 1
                    if f0[j + 1] < f0[j]:
                        level0 = s + 1
                    else:
                        level0 = s + 2
                    ipar[nboxes], level[nboxes], ichild[nboxes], f[0, nboxes] = MCSUtils().genbox(par, level0,
                                                                                                  -nchild, f0[j + 1])
                    record = MCSUtils().updtrec(nboxes, level[nboxes], f[0, :], record)
                else:
                    x[i] = x0[i, j + 1]
                    nbasket = nbasket + 1
                    if (len(xmin) == nbasket):
                        xmin.append(copy.deepcopy(x))
                        fmi.append(f0[j + 1])
                    else:
                        xmin[nbasket] = copy.deepcopy(x)
                        fmi[nbasket] = f0[j + 1]
            if x0[i, L[i]] < v[i]:
                nchild = nchild + 1
                nboxes = nboxes + 1
                ipar[nboxes], level[nboxes], ichild[nboxes], f[0, nboxes] = MCSUtils().genbox(par, s + 1,
                                                                                              -nchild, f0[L[i]])
                record = MCSUtils().updtrec(nboxes, level[nboxes], f[0, :], record)
        else:
            for j in range(L[i] + 1):
                x[i] = x0[i, j]
                nbasket = nbasket + 1
                if (len(xmin) == nbasket):
                    xmin.append(copy.deepcopy(x))
                    fmi.append(f0[j])
                else:
                    xmin[nbasket] = copy.deepcopy(x)
                    fmi[nbasket] = f0[j]
        return (xbest, fbest, f0, xmin, fmi, ipar, level, ichild, f, flag, ncall,
                record, nboxes, nbasket, nsweepbest, nsweep)

    def split(self, i: int, s: int, smax: int, par: int, n0: int, u: List[int], v: List[int], x: NDArray[np.float64], y: NDArray[np.float64],
              x1: NDArray[np.float64], x2: NDArray[np.float64], z: NDArray[np.float64], xmin: NDArray[np.float64], fmi: NDArray[np.float64],
              ipar: NDArray[np.int32], level: NDArray[np.int32], ichild: NDArray[np.int32], f: NDArray[np.float64], xbest: NDArray[np.float64],
              fbest: NDArray[np.float64], stop: List[int], prt: int, record: NDArray[np.int32],
              nboxes: int, nbasket: int, nsweepbest: int, nsweep: int, stopping_actions: int):
        """
        Split Function
        """
        ncall = 0
        flag = 1
        x[i] = z[1]
        policy = self.get_policy(x, L=stopping_actions)
        avg_metrics = self.eval_theta(policy=policy,
                                      max_steps=self.experiment_config.hparams[
                                          agents_constants.COMMON.MAX_ENV_STEPS].value)
        f[1, par] = round(avg_metrics[env_constants.ENV_METRICS.RETURN], 3)
        ncall = ncall + 1
        if f[1, par] < fbest:
            fbest = copy.deepcopy(f[1, par])
            xbest = copy.deepcopy(x)
            nsweepbest = copy.deepcopy(nsweep)
            if stop[0] > 0 and stop[0] < 1:
                flag = MCSUtils().chrelerr(fbest, stop)
            elif stop[0] == 0:
                flag = MCSUtils().chvtr(fbest, stop[2])

            if not flag:
                return (xbest, fbest, xmin, fmi, ipar, level, ichild, f,
                        flag, ncall, record, nboxes, nbasket, nsweepbest, nsweep)

        if s + 1 < smax:
            if f[0, par] <= f[1, par]:
                nboxes = nboxes + 1
                ipar[nboxes], level[nboxes], ichild[nboxes], f[0, nboxes] = MCSUtils().genbox(par, s + 1,
                                                                                              1, f[0, par])
                record = MCSUtils().updtrec(nboxes, level[nboxes], f[0, :], record)
                if s + 2 < smax:
                    nboxes = nboxes + 1
                    ipar[nboxes], level[nboxes], ichild[nboxes], f[0, nboxes] = MCSUtils().genbox(par, s + 2,
                                                                                                  2, f[1, par])
                    record = MCSUtils().updtrec(nboxes, level[nboxes], f[0, :], record)
                else:
                    x[i] = z[1]
                    nbasket = nbasket + 1
                    if (len(xmin) == nbasket):
                        xmin.append(copy.deepcopy(x))
                        fmi.append(f[1, par])
                    else:
                        xmin[nbasket] = copy.deepcopy(x)
                        fmi[nbasket] = f[1, par]
            else:
                if s + 2 < smax:
                    nboxes = nboxes + 1
                    ipar[nboxes], level[nboxes], ichild[nboxes], f[0, nboxes] = MCSUtils().genbox(par, s + 2,
                                                                                                  1, f[0, par])
                    record = MCSUtils().updtrec(nboxes, level[nboxes], f[0, :], record)
                else:
                    x[i] = z[0]
                    nbasket = nbasket + 1
                    if (len(xmin) == nbasket):
                        xmin.append(copy.deepcopy(x))
                        fmi.append(f[0, par])
                    else:
                        xmin[nbasket] = copy.deepcopy(x)
                        fmi[nbasket] = f[0, par]
                nboxes = nboxes + 1
                ipar[nboxes], level[nboxes], ichild[nboxes], f[0, nboxes] = MCSUtils().genbox(par, s + 1,
                                                                                              2, f[1, par])
                record = MCSUtils().updtrec(nboxes, level[nboxes], f[0, :], record)
            if z[1] != y[i]:
                if abs(z[1] - y[i]) > abs(z[1] - z[0]) * (3 - np.sqrt(5)) * 0.5:
                    nboxes = nboxes + 1
                    ipar[nboxes], level[nboxes], ichild[nboxes], f[0, nboxes] = MCSUtils().genbox(par, s + 1,
                                                                                                  3, f[1, par])
                    record = MCSUtils().updtrec(nboxes, level[nboxes], f[0, :], record)
                else:
                    if s + 2 < smax:
                        nboxes = nboxes + 1
                        ipar[nboxes], level[nboxes], ichild[nboxes], f[0, nboxes] = MCSUtils().genbox(par, s + 2,
                                                                                                      3, f[1, par])
                        record = MCSUtils().updtrec(nboxes, level[nboxes], f[0, :], record)
                    else:
                        x[i] = z[1]
                        nbasket = nbasket + 1
                        if (len(xmin) == nbasket):
                            xmin.append(copy.deepcopy(x))
                            fmi.append(copy.deepcopy(f[1, par]))
                        else:
                            xmin[nbasket] = copy.deepcopy(x)
                            fmi[nbasket] = f[1, par]

        else:
            xi1 = copy.deepcopy(x)
            xi2 = copy.deepcopy(x)

            xi1[i] = z[0]
            nbasket = nbasket + 1
            if (len(xmin) == nbasket):
                xmin.append(xi1)
                fmi.append(f[0, par])
            else:
                xmin[nbasket] = xi1
                fmi[nbasket] = f[0, par]

            xi2[i] = z[1]
            nbasket = nbasket + 1
            if (len(xmin) == nbasket):
                xmin.append(xi2)
                fmi.append(f[1, par])
            else:
                xmin[nbasket] = xi2
                fmi[nbasket] = f[1, par]
        return (xbest, fbest, xmin, fmi, ipar, level, ichild, f, flag,
                ncall, record, nboxes, nbasket, nsweepbest, nsweep)

    def basket(self, x, f, xmin, fmi, xbest, fbest, stop, nbasket, nsweep, nsweepbest, stopping_actions):
        loc = 1
        flag = 1
        ncall = 0
        if not nbasket:
            return xbest, fbest, xmin, fmi, x, f, loc, flag, ncall, nsweep, nsweepbest
        dist = np.zeros(nbasket + 1)
        for k in range(len(dist)):
            dist[k] = np.linalg.norm(np.subtract(x, xmin[k]))

        # dist1 = np.sort(dist)
        ind = np.argsort(dist)

        for k in range(nbasket + 1):
            i = ind[k]
            if fmi[i] <= f:
                p = xmin[i] - x

                y1 = x + 1 / 3 * p
                policy = self.get_policy(y1, L=stopping_actions)
                avg_metrics = self.eval_theta(policy=policy,
                                              max_steps=self.experiment_config.hparams[
                                                  agents_constants.COMMON.MAX_ENV_STEPS].value)
                f1 = round(avg_metrics[env_constants.ENV_METRICS.RETURN], 3)

                ncall = ncall + 1
                if f1 <= f:
                    y2 = x + 2 / 3 * p
                    policy = self.get_policy(y2, L=stopping_actions)
                    avg_metrics = self.eval_theta(policy=policy,
                                                  max_steps=self.experiment_config.hparams[
                                                      agents_constants.COMMON.MAX_ENV_STEPS].value)
                    f2 = round(avg_metrics[env_constants.ENV_METRICS.RETURN], 3)
                    ncall = ncall + 1
                    if f2 > max(f1, fmi[i]):
                        if f1 < f:
                            x = y1
                            f = f1
                            if f < fbest:
                                fbest = f
                                xbest = copy.deepcopy(x)
                                nsweepbest = nsweep
                                if stop[0] > 0 and stop[0] < 1:
                                    flag = MCSUtils().chrelerr(fbest, stop)
                                elif stop[0] == 0:
                                    flag = MCSUtils().chvtr(fbest, stop[1])
                                if not flag:
                                    return (
                                        xbest,
                                        fbest,
                                        xmin,
                                        fmi,
                                        x,
                                        f,
                                        loc,
                                        flag,
                                        ncall,
                                        nsweep,
                                        nsweepbest,
                                    )
                    else:
                        if f1 < min(f2, fmi[i]):
                            f = f1
                            x = copy.deepcopy(y1)
                            if f < fbest:
                                fbest = f
                                xbest = copy.deepcopy(x)
                                nsweepbest = nsweep
                                if stop[0] > 0 and stop[0] < 1:
                                    flag = MCSUtils().chrelerr(fbest, stop)
                                elif stop[0] == 0:
                                    flag = MCSUtils().chvtr(fbest, stop[1])
                                if not flag:
                                    return (
                                        xbest,
                                        fbest,
                                        xmin,
                                        fmi,
                                        x,
                                        f,
                                        loc,
                                        flag,
                                        ncall,
                                        nsweep,
                                        nsweepbest,
                                    )
                            elif f2 < min(f1, fmi[i]):
                                f = f2
                                x = copy.deepcopy(y2)
                                if f < fbest:
                                    fbest = f
                                    xbest = copy.deepcopy(x)
                                    nsweepbest = nsweep
                                    if stop[0] > 0 and stop[0] < 1:
                                        flag = MCSUtils().chrelerr(fbest, stop)
                                    elif stop[0] == 0:
                                        flag = MCSUtils().chvtr(fbest, stop[1])
                                    if not flag:
                                        return (
                                            xbest,
                                            fbest,
                                            xmin,
                                            fmi,
                                            x,
                                            f,
                                            loc,
                                            flag,
                                            ncall,
                                            nsweep,
                                            nsweepbest,
                                        )
                            else:
                                loc = 0
                                break

        return xbest, fbest, xmin, fmi, x, f, loc, flag, ncall, nsweep, nsweepbest

    def lsearch(self, x: NDArray[np.float64], f: NDArray[np.float64], f0: NDArray[np.float64], u: List[int], v: List[int],
                nf: int, stop: List[int], maxstep: int, gamma: float, hess: NDArray[np.float64], nsweep: int,
                nsweepbest: int, stopping_actions: int, eps: float):
        ncall = 0
        n = len(x)
        x0 = np.asarray([min(max(u[i], 0), v[i]) for i in range(len(u))])
        flag = 1
        eps0 = 0.001
        nloc = 1
        small = 0.1
        smaxls = 15
        
        xmin, fmi, g, G, nfcsearch = self.csearch(x, f, u, v, hess,
                                                  stopping_actions, eps)

        xmin = [max(u[i], min(xmin[i], v[i])) for i in range(n)]
        ncall = ncall + nfcsearch
        xold = copy.deepcopy(xmin)
        fold = copy.deepcopy(fmi)

        if stop[0] > 0 and stop[0] < 1:
            flag = MCSUtils().chrelerr(fmi, stop)
        elif stop[0] == 0:
            flag = MCSUtils().chvtr(fmi, stop[1])
        if not flag:
            return xmin, fmi, ncall, flag, nsweep, nsweepbest

        d = np.asarray([min(min(xmin[i] - u[i], v[i] - xmin[i]), 0.25 * (1 + abs(x[i] - x0[i]))) for i in range(n)])
        p, _, _ = LSUtils().minq(fmi, g, G, -d, d, 0)

        x = [max(u[i], min(xmin[i] + p[i], v[i])) for i in range(n)]
        p = np.subtract(x, xmin)
        if np.linalg.norm(p):
            policy = self.get_policy(x, L=stopping_actions)
            avg_metrics = self.eval_theta(policy=policy,
                                          max_steps=self.experiment_config.hparams[
                                              agents_constants.COMMON.MAX_ENV_STEPS].value)
            f1 = round(avg_metrics[env_constants.ENV_METRICS.RETURN], 3)
            ncall = ncall + 1
            alist = [0, 1]
            flist = [fmi, f1]
            fpred = fmi + np.dot(g.T, p) + np.dot(0.5, np.dot(p.T, np.dot(G, p)))
            alist, flist, nfls = self.gls(u, v, xmin, p, alist,
                                          flist, nloc, small, smaxls, stopping_actions)
            ncall = ncall + nfls

            i = np.argmin(flist)
            fminew = min(flist)
            if fminew == fmi:
                i = [i for i in range(len(alist)) if not alist[i]][0]
            else:
                fmi = copy.deepcopy(fminew)
            
            xmin = xmin + np.dot(alist[i], p)
            xmin = np.asarray([max(u[i], min(xmin[i], v[i])) for i in range(n)])
            gain = f - fmi
            
            if stop[0] > 0 and stop[0] < 1:
                flag = MCSUtils().chrelerr(fmi, stop)
            elif stop[0] == 0:
                flag = MCSUtils().chvtr(fmi, stop[1])

            if not flag:
                return xmin, fmi, ncall, flag, nsweep, nsweepbest

            if fold == fmi:
                r = 0
            elif fold == fpred:
                r = 0.5
            else:
                r = (fold - fmi) / (fold - fpred)
        else:
            gain = f - fmi
            r = 0

        diag = 0
        ind = [i for i in range(n) if (u[i] < xmin[i] and xmin[i] < v[i])]
        b = np.dot(np.abs(g).T, [max(abs(xmin[i]), abs(xold[i])) for i in range(len(xmin))])
        nstep = 0
        while (ncall < nf) and (nstep < maxstep) and ((diag or len(ind) < n) or
                                                      (stop[0] == 0 and fmi - gain <= stop[1]) or
                                                      (b >= gamma * (f0 - f) and gain > 0)):
            nstep = nstep + 1
            delta = [abs(xmin[i]) * eps ** (1 / 3) for i in range(len(xmin))]
            j = [inx for inx in range(len(delta)) if (not delta[inx])]
            if len(j) != 0:
                for inx in j:
                    delta[inx] = eps ** (1 / 3) * 1
                    
            x1, x2 = MCSUtils().neighbor(xmin, delta, u, v)
            f = copy.deepcopy(fmi)

            if len(ind) < n and (b < gamma * (f0 - f) or (not gain)):
                ind1 = [i for i in range(len(u)) if (xmin[i] == u[i] or xmin[i] == v[i])]
                for k in range(len(ind1)):
                    i = ind1[k]
                    x = copy.deepcopy(xmin)
                    if xmin[i] == u[i]:
                        x[i] = x2[i]
                    else:
                        x[i] = x1[i]
                    policy = self.get_policy(x, L=stopping_actions)
                    avg_metrics = self.eval_theta(policy=policy,
                                                  max_steps=self.experiment_config.hparams[
                                                      agents_constants.COMMON.MAX_ENV_STEPS].value)
                    f1 = round(avg_metrics[env_constants.ENV_METRICS.RETURN], 3)

                    ncall = ncall + 1

                    if f1 < fmi:
                        alist = [0, x[i], -xmin[i]]
                        flist = [fmi, f1]
                        p = np.zeros(n)
                        p[i] = 1
                        alist, flist, nfls = self.gls(u, v, xmin, p, alist,
                                                      flist, nloc, small, 6, stopping_actions)
                        ncall = ncall + nfls
                        j = np.argmin(flist)
                        fminew = min(flist)
                        if fminew == fmi:
                            j = [inx for inx in range(len(alist)) if (not alist[inx])][0]
                        else:
                            fmi = fminew
                        xmin[i] = xmin[i] + alist[j]
                    else:
                        ind1[k] = -1

                xmin = np.asarray([max(u[inx], min(xmin[inx], v[inx])) for inx in range(len(xmin))])
                if not sum(ind1):
                    break

                for inx in range(len(delta)):
                    delta[inx] = abs(xmin[inx]) * eps ** (1 / 3)
                j = [inx for inx in range(len(delta)) if (not delta[inx])]
                if len(j) != 0:
                    for inx in j:
                        delta[inx] = eps ** (1 / 3) * 1
                x1, x2 = MCSUtils().neighbor(xmin, delta, u, v)

            if abs(r - 1) > 0.25 or (not gain) or (b < gamma * (f0 - f)):
                xmin, fmi, g, G, x1, x2, nftriple = self.triple(xmin, fmi, x1, x2, u, v, hess, 0,
                                                                stopping_actions, setG=True)
                ncall = ncall + nftriple
                diag = 0
            else:
                xmin, fmi, g, G, x1, x2, nftriple = self.triple(xmin, fmi, x1, x2, u, v,
                                                                hess, G, stopping_actions)
                ncall = ncall + nftriple
                diag = 1
            xold = copy.deepcopy(xmin)
            fold = copy.deepcopy(fmi)

            if stop[0] > 0 and stop[0] < 1:
                flag = MCSUtils().chrelerr(fmi, stop)
            elif stop[0] == 0:
                flag = MCSUtils().chvtr(fmi, stop[1])

            if not flag:
                return xmin, fmi, ncall, flag, nsweep, nsweepbest
            if r < 0.25:
                d = 0.5 * d
            elif r > 0.75:
                d = 2 * d

            minusd = np.asarray([max(-d[jnx], u[jnx] - xmin[jnx]) for jnx in range(len(xmin))])
            mind = np.asarray([min(d[jnx], v[jnx] - xmin[jnx]) for jnx in range(len(xmin))])
            p, _, _ = LSUtils().minq(fmi, g, G, minusd, mind, 0)

            if not (np.linalg.norm(p)) and (not diag) and (len(ind) == n):
                break
            if np.linalg.norm(p):
                fpred = fmi + np.dot(g.T, p) + np.dot(0.5, np.dot(p.T, np.dot(G, p)))
                x = copy.deepcopy(xmin + p)
                policy = self.get_policy(x, L=stopping_actions)
                avg_metrics = self.eval_theta(policy=policy,
                                              max_steps=self.experiment_config.hparams[
                                                  agents_constants.COMMON.MAX_ENV_STEPS].value)
                f1 = round(avg_metrics[env_constants.ENV_METRICS.RETURN], 3)
                ncall = ncall + 1
                alist = [0, 1]
                flist = [fmi, f1]
                alist, flist, nfls = self.gls(u, v, xmin, p, alist,
                                              flist, nloc, small, smaxls, stopping_actions)
                ncall = ncall + nfls
                argmin = np.argmin(flist)
                fmi = min(flist)
                xmin = [xmin[jnx] + alist[argmin] * p[jnx] for jnx in range(len(xmin))]
                xmin = np.asarray([max(u[jnx], min(xmin[jnx], v[jnx])) for jnx in range(len(xmin))])
                if stop[0] > 0 and stop[0] < 1:
                    flag = MCSUtils().chrelerr(fmi, stop)
                elif stop[0] == 0:
                    flag = MCSUtils().chvtr(fmi, stop[1])
                if not flag:
                    return xmin, fmi, ncall, flag, nsweep, nsweepbest

                gain = f - fmi
                if fold == fmi:
                    r = 0
                elif fold == fpred:
                    r = 0.5
                else:
                    r = (fold - fmi) / (fold - fpred)
                if fmi < fold:
                    fac = abs(1 - 1 / r)
                    eps0 = max(eps, min(fac * eps0, 0.001))
                else:
                    eps0 = 0.001
            else:
                gain = f - fmi
                if (not gain):
                    eps0 = 0.001
                    fac = np.Inf
                    r = 0
            ind = [inx for inx in range(len(u)) if (u[inx] < xmin[inx] and xmin[inx] < v[inx])]
            b = np.dot(np.abs(g).T, [max(abs(xmin[inx]), abs(xold[inx])) for inx in range(len(xmin))])
        return xmin, fmi, ncall, flag, nsweep, nsweepbest

    def basket1(self, x: NDArray[np.float64], f: NDArray[np.float64], xmin: NDArray[np.int32], fmi: NDArray[np.float64],
                xbest: NDArray[np.float64], fbest: NDArray[np.float64], stop: List[int], nbasket: int, nsweep: int,
                nsweepbest: int, stopping_actions: int):
        loc = 1
        flag = 1
        ncall = 0
        if not nbasket:
            return xbest, fbest, xmin, fmi, loc, flag, ncall, nsweep, nsweepbest
        dist = np.zeros(nbasket + 1)
        for k in range(len(dist)):
            dist[k] = np.linalg.norm(np.subtract(x, xmin[k]))
        # dist1 = np.sort(dist)
        ind = np.argsort(dist)

        for k in range(nbasket + 1):
            i = ind[k]
            p = xmin[i] - x
            y1 = x + 1 / 3 * p
            policy = self.get_policy(y1, L=stopping_actions)
            avg_metrics = self.eval_theta(policy=policy,
                                          max_steps=self.experiment_config.hparams[
                                              agents_constants.COMMON.MAX_ENV_STEPS].value)
            f1 = round(avg_metrics[env_constants.ENV_METRICS.RETURN], 3)
            ncall = ncall + 1
            if f1 <= max(fmi[i], f):
                y2 = x + 2 / 3 * p
                policy = self.get_policy(y2, L=stopping_actions)
                avg_metrics = self.eval_theta(policy=policy,
                                              max_steps=self.experiment_config.hparams[
                                                  agents_constants.COMMON.MAX_ENV_STEPS].value)
                f2 = round(avg_metrics[env_constants.ENV_METRICS.RETURN], 3)
                ncall = ncall + 1
                if f2 <= max(f1, fmi[i]):
                    if f < min(min(f1, f2), fmi[i]):
                        fmi[i] = f
                        xmin[i] = copy.deepcopy(x)
                        if fmi[i] < fbest:
                            fbest = copy.deepcopy(fmi[i])
                            xbest = copy.deepcopy(xmin[i])
                            nsweepbest = nsweep
                            if stop[0] > 0 and stop[0] < 1:
                                flag = MCSUtils().chrelerr(fbest, stop)
                            elif stop[0] == 0:
                                flag = MCSUtils().chvtr(fbest, stop[1])
                            if not flag:
                                return (
                                    xbest,
                                    fbest,
                                    xmin,
                                    fmi,
                                    loc,
                                    flag,
                                    ncall,
                                    nsweep,
                                    nsweepbest,
                                )
                        # end fmi[i] < fbest:
                        loc = 0
                        break
                    elif f1 < min(min(f, f2), fmi[i]):
                        fmi[i] = f1
                        xmin[i] = copy.deepcopy(y1)
                        if fmi[i] < fbest:
                            fbest = copy.deepcopy(fmi[i])
                            xbest = copy.deepcopy(xmin[i])
                            nsweepbest = copy.deepcopy(nsweep)

                            if stop[0] > 0 and stop[0] < 1:
                                flag = MCSUtils().chrelerr(fbest, stop)
                            elif stop[0] == 0:
                                flag = MCSUtils().chvtr(fbest, stop[1])
                            if not flag:
                                return (
                                    xbest,
                                    fbest,
                                    xmin,
                                    fmi,
                                    loc,
                                    flag,
                                    ncall,
                                    nsweep,
                                    nsweepbest,
                                )
                        # end fmi[i] < fbest: elif
                        loc = 0
                        break
                    elif f2 < min(min(f, f1), fmi[i]):
                        fmi[i] = f2
                        xmin[i] = copy.deepcopy(y2)
                        if fmi[i] < fbest:
                            fbest = copy.deepcopy(fmi[i])
                            xbest = copy.deepcopy(xmin[i])
                            nsweepbest = nsweep
                            if stop[0] > 0 and stop[0] < 1:
                                flag = MCSUtils().chrelerr(fbest, stop)
                            elif stop[0] == 0:
                                flag = MCSUtils().chvtr(fbest, stop[1])
                            if not flag:
                                return (
                                    xbest,
                                    fbest,
                                    xmin,
                                    fmi,
                                    loc,
                                    flag,
                                    ncall,
                                    nsweep,
                                    nsweepbest,
                                )
                        loc = 0
                        break
                    else:
                        loc = 0
                        break
        return xbest, fbest, xmin, fmi, loc, flag, ncall, nsweep, nsweepbest

    def csearch(self, x: NDArray[np.float64], f: NDArray[np.float64], u: List[int], v: List[int], hess: NDArray[np.float64],
                stopping_actions: int, eps: float):
        n = len(x)
        x = [min(v[i], max(x[i], u[i])) for i in range(len(x))]

        nfcsearch = 0
        smaxls = 6
        small = 0.1
        nloc = 1
        hess = np.ones((n, n))
        xmin = copy.deepcopy(x)
        fmi = copy.deepcopy(f)
        xminnew = copy.deepcopy(xmin)
        fminew = copy.deepcopy(fmi)
        g = np.zeros(n)
        # ind0 = []
        x1 = np.zeros(n)
        x2 = np.zeros(n)
        G = np.zeros((n, n))

        for i in range(n):
            p = np.zeros(n)
            p[i] = 1
            if xmin[i]:
                delta = eps ** (1 / 3) * abs(xmin[i])
            else:
                delta = eps ** (1 / 3)
            linesearch = True
            if xmin[i] <= u[i]:
                policy = self.get_policy(xmin + delta * p, L=stopping_actions)
                avg_metrics = self.eval_theta(policy=policy,
                                              max_steps=self.experiment_config.hparams[
                                                  agents_constants.COMMON.MAX_ENV_STEPS].value)
                f1 = round(avg_metrics[env_constants.ENV_METRICS.RETURN], 3)
                nfcsearch = nfcsearch + 1
                if f1 >= fmi:
                    policy = self.get_policy(xmin + 2 * delta * p, L=stopping_actions)
                    avg_metrics = self.eval_theta(policy=policy,
                                                  max_steps=self.experiment_config.hparams[
                                                      agents_constants.COMMON.MAX_ENV_STEPS].value)
                    f2 = round(avg_metrics[env_constants.ENV_METRICS.RETURN], 3)
                    # fcsearch = nfcsearch + 1
                    x1[i] = xmin[i] + delta
                    x2[i] = xmin[i] + 2 * delta
                    if f2 >= fmi:
                        xminnew[i] = xmin[i]
                        fminew = fmi
                    else:
                        xminnew[i] = x2[i]
                        fminew = copy.deepcopy(f2)
                    linesearch = False
                else:
                    alist = [0, delta]
                    flist = [fmi, f1]
            elif xmin[i] >= v[i]:
                policy = self.get_policy(xmin - delta * p, L=stopping_actions)
                avg_metrics = self.eval_theta(policy=policy,
                                              max_steps=self.experiment_config.hparams[
                                                  agents_constants.COMMON.MAX_ENV_STEPS].value)
                f1 = round(avg_metrics[env_constants.ENV_METRICS.RETURN], 3)
                nfcsearch = nfcsearch + 1
                if f1 >= fmi:
                    policy = self.get_policy(xmin - 2 * delta * p, L=stopping_actions)
                    avg_metrics = self.eval_theta(policy=policy,
                                                  max_steps=self.experiment_config.hparams[
                                                      agents_constants.COMMON.MAX_ENV_STEPS].value)
                    f2 = round(avg_metrics[env_constants.ENV_METRICS.RETURN], 3)
                    nfcsearch = nfcsearch + 1
                    x1[i] = xmin[i] - delta
                    x2[i] = xmin[i] - 2 * delta
                    if f2 >= fmi:
                        xminnew[i] = xmin[i]
                        fminew = fmi
                    else:
                        xminnew[i] = x2[i]
                        fminew = f2
                    linesearch = False
                else:
                    alist = [0, -delta]
                    flist = [fmi, f1]
            else:
                alist = 0
                flist = fmi
            
            if linesearch:
                alist, flist, nfls = self.gls(u, v, xmin, p, alist, flist, nloc,
                                              small, smaxls, stopping_actions)
                nfcsearch = nfcsearch + nfls
                j = np.argmin(flist)
                fminew = min(flist)

                if fminew == fmi:
                    j = [inx for inx in range(len(alist)) if not alist[inx]][0]

                ind = [inx for inx in range(len(alist)) if abs(alist[inx] - alist[j]) < delta]
                ind1 = [inx for inx in range(len(ind)) if ind[inx] == j]
                for inx in ind1:
                    del ind[inx]

                for inx in ind:
                    del alist[inx]
                    del flist[inx]

                j = np.argmin(flist)
                fminew = min(flist)
                xminnew[i] = xmin[i] + alist[j]
                if i == 0 or not alist[j]:
                    if j == 0:
                        x1[i] = xmin[i] + alist[1]
                        f1 = flist[1]
                        x2[i] = xmin[i] + alist[2]
                        f2 = flist[2]
                    elif j == len(alist) - 1:
                        x1[i] = xmin[i] + alist[j - 1]
                        f1 = flist[j - 1]
                        x2[i] = xmin[i] + alist[j - 2]
                        f2 = flist[j - 2]
                    else:
                        x1[i] = xmin[i] + alist[j - 1]
                        f1 = flist[j - 1]
                        x2[i] = xmin[i] + alist[j + 1]
                        f2 = flist[j + 1]
                    xmin[i] = xminnew[i]
                    fmi = copy.deepcopy(fminew)
                else:
                    x1[i] = xminnew[i]
                    f1 = copy.deepcopy(fminew)
                    if xmin[i] < x1[i] and j < len(alist) - 1:
                        x2[i] = xmin[i] + alist[j + 1]
                        f2 = flist[j + 1]
                    elif j == 0:
                        if alist[j + 1]:
                            x2[i] = xmin[i] + alist[j + 1]
                            f2 = flist[j + 1]
                        else:
                            x2[i] = xmin[i] + alist[j + 2]
                            f2 = flist[j + 2]
                    elif alist[j - 1]:
                        x2[i] = xmin[i] + alist[j - 1]
                        f2 = flist[j - 1]
                    else:
                        x2[i] = xmin[i] + alist[j - 2]
                        f2 = flist[j - 2]
            g[i], G[i, i] = MCSUtils().polint1([xmin[i], x1[i], x2[i]], [fmi, f1, f2])
            x = copy.deepcopy(xmin)
            k1 = -1
            if f1 <= f2:
                x[i] = x1[i]
            else:
                x[i] = x2[i]
            for k in range(i):
                if hess[i, k]:
                    q1 = fmi + g[k] * (x1[k] - xmin[k]) + 0.5 * G[k, k] * (x1[k] - xmin[k]) ** 2
                    q2 = fmi + g[k] * (x2[k] - xmin[k]) + 0.5 * G[k, k] * (x2[k] - xmin[k]) ** 2
                    if q1 <= q2:
                        x[k] = x1[k]
                    else:
                        x[k] = x2[k]
                    policy = self.get_policy(x, L=stopping_actions)
                    avg_metrics = self.eval_theta(policy=policy,
                                                  max_steps=self.experiment_config.hparams[
                                                      agents_constants.COMMON.MAX_ENV_STEPS].value)
                    f12 = round(avg_metrics[env_constants.ENV_METRICS.RETURN], 3)
                    nfcsearch = nfcsearch + 1
                    G[i, k] = MCSUtils().hessian(i, k, x, xmin, f12, fmi, g, G)
                    G[k, i] = G[i, k]
                    if f12 < fminew:
                        fminew = f12
                        xminnew = copy.deepcopy(x)
                        k1 = k
                    x[k] = xmin[k]
                else:
                    G[i, k] = 0
                    G[k, i] = 0
            if fminew <= fmi:
                if x1[i] == xminnew[i]:
                    x1[i] = xmin[i]
                elif x2[i] == xminnew[i]:
                    x2[i] = xmin[i]
                if k1 > -1:
                    if xminnew[k1] == x1[k1]:
                        x1[k1] = xmin[k1]
                    elif xminnew[k1] == x2[k1]:
                        x2[k1] = xmin[k1]

                for k in range(i + 1):
                    g[k] = g[k] + G[i, k] * (xminnew[i] - xmin[i])
                    if k1 > -1:
                        g[k] = g[k] + G[k1, k] * (xminnew[k1] - xmin[k1])
            xmin = copy.deepcopy(xminnew)
            fmi = copy.deepcopy(fminew)
        return xmin, fmi, g, G, nfcsearch

    def gls(self, xl: List[int], xu: List[int], x: List[int], p: NDArray[np.int32], alist: List[int], flist: List[float],
            nloc: int, small: float, smax: int, stopping_actions: int, prt: int=2):
        '''
        Global line search main function
        :param func: funciton name which is subjected to optimization
        :param xl: lower bound
        :param xu: upper bound
        :param x: starting point
        :param p: search direction [1 or -1 ? need to check]
        :param alist: list of known steps
        :param flist: funciton values of known steps
        :param nloc: best local optimizal
        :param small: tollarance values
        :param smax: search list size
        :param prt: print - unsued in this implementation so far
        :return: search list,function values,number of fucntion evaluation
        '''
        if np.isscalar(alist):
            alist = [alist]
            flist = [flist]
        # if isinstance(alist, list): TODO: this should work
        # if isinstance(flist, list): TODO: this should work
        if type(alist) != list:
            alist = alist.tolist()
        if type(flist) != list:
            flist = flist.tolist()

        short = 0.381966
        sinit = len(alist)

        bend = 0
        xl, xu, x, p, amin, amax, scale = GLSUtils().lsrange(xl, xu, x, p, prt, bend)
        alist, flist, alp, alp1, alp2, falp = self.lsinit(x, p, alist, flist, amin, amax, scale, stopping_actions)
        alist, flist, abest, fbest, fmed, up, down, monotone, minima, nmin, unitlen, s = GLSUtils().lssort(alist, flist)
        nf = s - sinit

        while s < min(5, smax):
            if nloc == 1:
                (alist, flist, abest, fbest, fmed, up, down, monotone,
                 minima, nmin, unitlen, s, alp, fac) = self.lspar(nloc, small, sinit, short, x, p, alist, flist,
                                                                  amin, amax, alp, abest, fbest, fmed, up, down,
                                                                  monotone, minima, nmin, unitlen, s, stopping_actions)

                if s > 3 and monotone and (abest == amin or abest == amax):
                    nf = s - sinit
                    return alist, flist, nf
            else:
                alist, flist, alp, fac = self.lsnew(nloc, small, sinit, short, x, p, s, alist, flist,
                                                    amin, amax, alp, abest, fmed, unitlen, stopping_actions)
                (alist, flist, abest, fbest, fmed, up, down, monotone,
                 minima, nmin, unitlen, s) = GLSUtils().lssort(alist, flist)
        saturated = 0
        if nmin == 1:
            if monotone and (abest == amin or abest == amax):
                nf = s - sinit
                return alist, flist, nf
            if s == 5:

                (alist, flist, amin, amax, alp, abest, fbest, fmed, up, down,
                 monotone, minima, nmin, unitlen, s, good, saturated) = self.lsquart(nloc, small, sinit,
                                                                                     short, x, p, alist,
                                                                                     flist, amin, amax, alp,
                                                                                     abest, fbest, fmed, up,
                                                                                     down, monotone, minima,
                                                                                     nmin, unitlen, s, saturated,
                                                                                     stopping_actions)
            (alist, flist, alp, abest, fbest, fmed, up, down, monotone,
             minima, nmin, unitlen, s) = self.lsdescent(x, p, alist, flist, alp,
                                                        abest, fbest, fmed, up, down,
                                                        monotone, minima, nmin, unitlen,
                                                        s, stopping_actions)
            convex = GLSUtils().lsconvex(alist, flist, nmin, s)
            if convex:
                nf = s - sinit
                return alist, flist, nf
        sold = 0

        while 1:
            (alist, flist, alp, abest, fbest, fmed, up, down, monotone,
             minima, nmin, unitlen, s) = self.lsdescent(x, p, alist, flist, alp, abest,
                                                        fbest, fmed, up, down, monotone,
                                                        minima, nmin, unitlen, s, stopping_actions)
            alp, saturated = GLSUtils().lssat(small, alist, flist, alp, amin, amax, s, saturated)
            if saturated or s == sold or s >= smax:
                if saturated:
                    no_print = 0
                if s == sold:
                    no_print = 0
                if s >= smax:
                    no_print = 0
                break
            sold = s
            nminold = nmin
            if not saturated and nloc > 1:

                (alist, flist, amin, amax, alp, abest, fbest, fmed, up, down, monotone,
                 minima, nmin, unitlen, s) = self.lssep(nloc, small,
                                                        sinit, short, x,
                                                        p, alist, flist, amin,
                                                        amax, alp, abest, fbest,
                                                        fmed, up, down, monotone,
                                                        minima, nmin, unitlen, s,
                                                        stopping_actions)
            (alist, flist, alp, abest, fbest, fmed, up, down, monotone,
             minima, nmin, unitlen, s, saturated) = self.lslocal(nloc, small, sinit, short,
                                                                 x, p, alist, flist, amin, amax,
                                                                 alp, abest, fbest, fmed, up,
                                                                 down, monotone, minima, nmin,
                                                                 unitlen, s, saturated, stopping_actions)
            if nmin > nminold:
                saturated = 0
        nf = s - sinit

        return alist, flist, nf

    def lsinit(self, x, p, alist, flist, amin, amax, scale, stopping_actions):
        '''
            Line search intilization
        '''
        alp = 0
        alp1 = 0
        alp2 = 0
        falp = 0
        
        if len(alist) == 0:
            # evaluate at absolutely smallest point
            alp = 0
            if amin > 0:
                alp = amin
            if amax < 0:
                alp = amax
            policy = self.get_policy(x + alp * p, L=stopping_actions)
            avg_metrics = self.eval_theta(policy=policy,
                                          max_steps=self.experiment_config.hparams[
                                              agents_constants.COMMON.MAX_ENV_STEPS].value)
            falp = round(avg_metrics[env_constants.ENV_METRICS.RETURN], 3)
            alist.append(alp)
            flist.append(falp)
        elif len(alist) == 1:
            # evaluate at absolutely smallest point
            alp = 0
            if amin > 0:
                alp = amin
            if amax < 0:
                alp = amax
            if alist[0] != alp:
                policy = self.get_policy(x + alp * p, L=stopping_actions)
                avg_metrics = self.eval_theta(policy=policy,
                                              max_steps=self.experiment_config.hparams[
                                                  agents_constants.COMMON.MAX_ENV_STEPS].value)
                falp = round(avg_metrics[env_constants.ENV_METRICS.RETURN], 3)
                alist.append(alp)
                flist.append(falp)

        aamin = min(alist)
        aamax = max(alist)
        if amin > aamin or amax < aamax:
            sys.exit('GLS Error: non-admissible step in alist')
        if aamax - aamin <= scale:
            alp1 = max(amin, min(- scale, amax))
            alp2 = max(amin, min(+ scale, amax))
            alp = np.Inf
            
            if aamin - alp1 >= alp2 - aamax:
                alp = alp1
            if alp2 - aamax >= aamin - alp1:
                alp = alp2
            if alp < aamin or alp > aamax:
                policy = self.get_policy(x + alp * p, L=stopping_actions)
                avg_metrics = self.eval_theta(policy=policy,
                                              max_steps=self.experiment_config.hparams[
                                                  agents_constants.COMMON.MAX_ENV_STEPS].value)
                falp = round(avg_metrics[env_constants.ENV_METRICS.RETURN], 3)
                alist.append(alp)
                flist.append(falp)
        if len(alist) == 1:
            sys.exit('GLS Error: lsinit bug: no second point found')

        return alist, flist, alp, alp1, alp2, falp

    def triple(self, x, f, x1, x2, u, v, hess, G, stopping_actions, setG=False):
        nf = 0
        n = len(x)
        g = np.zeros(n)
        nargin = 10
        if setG:
            nargin = 9
            G = np.zeros((n, n))
        
        ind = [i for i in range(n) if (u[i] < x[i] and x[i] < v[i])]
        ind1 = [i for i in range(n) if (x[i] <= u[i] or x[i] >= v[i])]
        
        for j in range(len(ind1)):
            g[ind1[j]] = 0
            for k in range(n):
                G[ind1[j], k] = 0
                G[k, ind1[j]] = 0
                
        if len(ind) <= 1:
            xtrip = copy.deepcopy(x)
            ftrip = copy.deepcopy(f)
            if len(ind) != 0:
                for i in ind:
                    g[i] = 1
                    G[i, i] = 1
            return xtrip, ftrip, g, G, x1, x2, nf
        
        if setG:
            G = np.zeros((n, n))
        xtrip = copy.deepcopy(x)
        ftrip = copy.deepcopy(f)
        xtripnew = copy.deepcopy(x)
        ftripnew = copy.deepcopy(f)
        for j in range(len(ind)):
            i = ind[j]
            x = copy.deepcopy(xtrip)
            f = copy.deepcopy(ftrip)
            
            x[i] = x1[i]

            policy = self.get_policy(x, L=stopping_actions)
            avg_metrics = self.eval_theta(policy=policy,
                                          max_steps=self.experiment_config.hparams[
                                              agents_constants.COMMON.MAX_ENV_STEPS].value)
            f1 = round(avg_metrics[env_constants.ENV_METRICS.RETURN], 3)
            x[i] = x2[i]
            policy = self.get_policy(x, L=stopping_actions)
            avg_metrics = self.eval_theta(policy=policy,
                                          max_steps=self.experiment_config.hparams[
                                              agents_constants.COMMON.MAX_ENV_STEPS].value)
            f2 = round(avg_metrics[env_constants.ENV_METRICS.RETURN], 3)
            nf = nf + 2
            g[i], G[i, i] = MCSUtils().polint1([xtrip[i], x1[i], x2[i]], [f, f1, f2])
            if f1 <= f2:
                if f1 < ftrip:
                    ftripnew = copy.deepcopy(f1)
                    xtripnew[i] = x1[i]
            else:
                if f2 < ftrip:
                    ftripnew = copy.deepcopy(f2)
                    xtripnew[i] = x2[i]
            
            if nargin < 10:
                k1 = -1
                if f1 <= f2:
                    x[i] = x1[i]
                else:
                    x[i] = x2[i]
                
                for k in range(i):
                    if hess[i, k]:
                        if xtrip[k] > u[k] and xtrip[k] < v[k] and (
                            len([m for m in range(len(ind)) if ind[m] == k]) != 0):
                            q1 = ftrip + g[k] * (x1[k] - xtrip[k]) + 0.5 * G[k, k] * (x1[k] - xtrip[k]) ** 2
                            q2 = ftrip + g[k] * (x2[k] - xtrip[k]) + 0.5 * G[k, k] * (x2[k] - xtrip[k]) ** 2
                            if q1 <= q2:
                                x[k] = x1[k]
                            else:
                                x[k] = x2[k]
                            policy = self.get_policy(x, L=stopping_actions)
                            avg_metrics = self.eval_theta(policy=policy,
                                                          max_steps=self.experiment_config.hparams[
                                                              agents_constants.COMMON.MAX_ENV_STEPS].value)
                            f12 = round(avg_metrics[env_constants.ENV_METRICS.RETURN], 3)
                            nf = nf + 1
                            G[i, k] = MCSUtils().hessian(i, k, x, xtrip, f12, ftrip, g, G)
                            G[k, i] = G[i, k]
                            if f12 < ftripnew:
                                ftripnew = copy.deepcopy(f12)
                                xtripnew = copy.deepcopy(x)
                                k1 = k
                            x[k] = xtrip[k]
                    else:
                        G[i, k] = 0
                        G[k, i] = 0

            if ftripnew < ftrip:
                if x1[i] == xtripnew[i]:
                    x1[i] = xtrip[i]
                else:
                    x2[i] = xtrip[i]
                if nargin < 10 and k1 > -1:
                    if xtripnew[k1] == x1[k1]:
                        x1[k1] = xtrip[k1]
                    else:
                        x2[k1] = xtrip[k1]
                for k in range(i + 1):
                    if (len([m for m in range(len(ind)) if ind[m] == k]) != 0):
                        g[k] = g[k] + G[i, k] * (xtripnew[i] - xtrip[i])
                        if nargin < 10 and k1 > -1:
                            g[k] = g[k] + G[k1, k] * (xtripnew[k1] - xtrip[k1])
                xtrip = copy.deepcopy(xtripnew)
                ftrip = copy.deepcopy(ftripnew)
        return xtrip, ftrip, g, G, x1, x2, nf

    def lspar(self, nloc, small, sinit, short, x, p, alist, flist, amin, amax, alp, abest, fbest,
              fmed, up, down, monotone, minima, nmin, unitlen, s, stopping_actions):
        cont = 1
        fac = short
        if s < 3:
            alist, flist, alp, fac = self.lsnew(nloc, small, sinit, short, x, p, s,
                                                alist, flist, amin, amax, alp,
                                                abest, fmed, unitlen, stopping_actions)
            cont = 0

        if cont:
            # fmin = min(flist)
            i = np.argmin(flist)
            if i <= 1:
                ind = [j for j in range(3)]
                ii = copy.deepcopy(i)
            elif i >= s - 2:
                ind = [j for j in range(s - 2 - 1, s)]
                ii = i - (s - 1) + 2
            else:
                ind = [j for j in range(ii - 1, i + 1)]
                ii = 2 - 1

            aa = [alist[j] for j in ind]
            ff = [flist[j] for j in ind]

            f12 = (ff[1] - ff[0]) / (aa[1] - aa[0])
            f23 = (ff[2] - ff[1]) / (aa[2] - aa[1])
            f123 = (f23 - f12) / (aa[2] - aa[0])
            if not (f123 > 0):
                alist, flist, alp, fac = self.lsnew(nloc, small, sinit, short, x, p, s, alist,
                                                    flist, amin, amax, alp, abest, fmed, unitlen,
                                                    stopping_actions)
                # alist,flist,abest,fbest,fmed,up,down,monotone,minima,nmin,unitlen,s = GLSUtils().lssort(alist,flist)
                cont = 0
        if cont:
            alp0 = 0.5 * (aa[1] + aa[2] - f23 / f123)
            alp = LSUtils().lsguard(alp0, alist, amax, amin, small)
            alptol = small * (aa[2] - aa[0])
            if f123 == np.Inf or min([abs(i - alp) for i in alist]) <= alptol:
                if ii == 0 or (ii == 1 and (aa[1] >= 0.5 * (aa[0] + aa[2]))):
                    alp = 0.5 * (aa[0] + aa[1])
                else:
                    alp = 0.5 * (aa[1] + aa[2])
            # else:
            #     np_print = alp0

            policy = self.get_policy(x + alp * p, L=stopping_actions)
            avg_metrics = self.eval_theta(policy=policy,
                                          max_steps=self.experiment_config.hparams[
                                              agents_constants.COMMON.MAX_ENV_STEPS].value)
            falp = round(avg_metrics[env_constants.ENV_METRICS.RETURN], 3)
            alist.append(alp)
            flist.append(falp)
        (alist, flist, abest, fbest, fmed, up, down, monotone,
         minima, nmin, unitlen, s) = GLSUtils().lssort(alist, flist)
        return alist, flist, abest, fbest, fmed, up, down, monotone, minima, nmin, unitlen, s, alp, fac

    def lsnew(self, nloc, small, sinit, short, x, p, s, alist, flist, amin,
              amax, alp, abest, fmed, unitlen, stopping_actions):
        if alist[0] <= amin:
            leftok = 0
        elif flist[0] >= max(fmed, flist[1]):
            leftok = (sinit == 1 or nloc > 1)
        else:
            leftok = 1
        if alist[s - 1] >= amax:
            rightok = 0
        elif flist[s - 1] >= max(fmed, flist[s - 2]):
            rightok = (sinit == 1 or nloc > 1)
        else:
            rightok = 1
        if sinit == 1:
            step = s - 1
        else:
            step = 1
        fac = short
        if leftok and (flist[0] < flist[s - 1] or (not rightok)):
            extra = 1
            al = alist[0] - (alist[0 + step] - alist[0]) / small
            alp = max(amin, al)
        elif rightok:
            # extra = 1
            au = alist[s - 1] + (alist[s - 1] - alist[s - 1 - step]) / small
            alp = min(au, amax)
        else:
            # extra = 0
            lenth = [i - j for i, j in zip(alist[1: s], alist[0: s - 1])]
            dist = [max(i, j, k) for i, j, k in zip([i - abest for i in alist[1: s]],
                                                    [abest - i for i in alist[0: s - 1]],
                                                    (unitlen * np.ones(s - 1)).tolist())]
            wid = [lenth[i] / dist[i] for i in range(len(lenth))]
            i = np.argmax(wid)
            wid = max(wid)
            alp, fac = LSUtils().lssplit(i, alist, flist, short)

        policy = self.get_policy(x + alp * p, L=stopping_actions)
        avg_metrics = self.eval_theta(policy=policy,
                                      max_steps=self.experiment_config.hparams[
                                          agents_constants.COMMON.MAX_ENV_STEPS].value)
        falp = round(avg_metrics[env_constants.ENV_METRICS.RETURN], 3)
        alist.append(alp)
        flist.append(falp)

        return alist, flist, alp, fac

    def lsdescent(self, x: NDArray[np.float64], p: NDArray[np.int32], alist: List[float], flist: List[int], alp: int,
                  abest: float, fbest: float, fmed: float, up: List[float], down: List[float], monotone: int, minima: List[int],
                  nmin: int, unitlen: float, s: int, stopping_actions: int):
        cont = max([i == 0 for i in alist])

        if cont:
            fbest = min(flist)
            i = np.argmin(flist)
            if alist[i] < 0:
                if alist[i] >= 4 * alist[i + 1]:
                    cont = 0
            elif alist[i] > 0:
                if alist[i] < 4 * alist[i - 1]:
                    cont = 0
            else:
                if i == 0:
                    fbest = flist[1]
                elif i == s - 1:
                    fbest = flist[s - 2]
                else:
                    fbest = min(flist[i - 1], flist[i + 1])
        if cont:
            if alist[i] != 0:
                alp = alist[i] / 3
            elif i == s - 1:
                alp = alist[s - 2] / 3
            elif i == 0:
                alp = alist[1] / 3
            else:
                if alist[i + 1] - alist[i] > alist[i] - alist[i - 1]:
                    alp = alist[i + 1] / 3
                else:
                    alp = alist[i - 1] / 3
            policy = self.get_policy(x + alp * p, L=stopping_actions)
            avg_metrics = self.eval_theta(policy=policy,
                                          max_steps=self.experiment_config.hparams[
                                              agents_constants.COMMON.MAX_ENV_STEPS].value)
            falp = round(avg_metrics[env_constants.ENV_METRICS.RETURN], 3)
            alist.append(alp)
            flist.append(falp)
            (alist, flist, abest, fbest, fmed, up, down, monotone,
             minima, nmin, unitlen, s) = GLSUtils().lssort(alist, flist)
        return (alist, flist, alp, abest, fbest, fmed, up, down,
                monotone, minima, nmin, unitlen, s)

    def lsquart(self, nloc: int, small: int, sinit: int, short: float, x: NDArray[np.float64], p: NDArray[np.int32],
                alist: List[float], flist: NDArray[np.float64], amin: float, amax: float, alp: float, abest: float, fbest: float,
                fmed: float, up: List[float], down: List[float],
                monotone: int, minima: List[int], nmin: int, unitlen: float, s: int,
                saturated: int, stopping_actions: int):

        if alist[0] == alist[1]:
            f12 = 0
        else:
            f12 = (flist[1] - flist[0]) / (alist[1] - alist[0])
        
        if alist[1] == alist[2]:
            f23 = 0
        else:
            f23 = (flist[2] - flist[1]) / (alist[2] - alist[1])

        if alist[2] == alist[3]:
            f34 = 0
        else:
            f34 = (flist[3] - flist[2]) / (alist[3] - alist[2])

        if alist[3] == alist[4]:
            f45 = 0
        else:
            f45 = (flist[4] - flist[3]) / (alist[4] - alist[3])

        f123 = (f23 - f12) / (alist[2] - alist[0])
        f234 = (f34 - f23) / (alist[3] - alist[1])
        f345 = (f45 - f34) / (alist[4] - alist[2])
        f1234 = (f234 - f123) / (alist[3] - alist[0])
        f2345 = (f345 - f234) / (alist[4] - alist[1])
        f12345 = (f2345 - f1234) / (alist[4] - alist[0])
        good = np.Inf
        if f12345 <= 0:

            good = 0
            (alist, flist, alp, abest, fbest, fmed, up, down, monotone,
             minima, nmin, unitlen, s, saturated) = self.lslocal(nloc, small, sinit,
                                                                 short, x, p, alist,
                                                                 flist, amin, amax, alp,
                                                                 abest, fbest, fmed, up,
                                                                 down, monotone, minima, nmin,
                                                                 unitlen, s, saturated, stopping_actions)
            quart = 0
        else:
            quart = 1

        if quart:
            c = np.zeros(len(alist))
            c[0] = f12345
            c[1] = f1234 + c[0] * (alist[2] - alist[0])
            c[2] = f234 + c[1] * (alist[2] - alist[3])
            c[1] = c[1] + c[0] * (alist[2] - alist[3])
            c[3] = f23 + c[2] * (alist[2] - alist[1])
            c[2] = c[2] + c[1] * (alist[2] - alist[1])
            c[1] = c[1] + c[0] * (alist[2] - alist[1])
            c[4] = flist[2]
            cmax = max(c)
            c = np.divide(c, cmax)
            hk = 4 * c[0]
            compmat = [[0, 0, - c[3]], [hk, 0, - 2 * c[2]], [0, hk, - 3 * c[1]]]
            ev = np.divide(np.linalg.eig(compmat)[0], hk)
            i = np.where(ev.imag == 0)

            if i[0].shape[0] == 1:
                alp = alist[2] + ev[i[0][0]]

            else:
                ev = np.sort(ev)
                alp1 = LSUtils().lsguard(alist[2] + ev[0], alist, amax, amin, small)
                alp2 = LSUtils().lsguard(alist[2] + ev[2], alist, amax, amin, small)
                f1 = cmax * LSUtils().quartic(c, alp1 - alist[2])
                f2 = cmax * LSUtils().quartic(c, alp2 - alist[2])

                if alp2 > alist[4] and f2 < max(flist):
                    alp = alp2
                elif alp1 < alist[0] and f1 < max(flist):
                    alp = alp1
                elif f2 <= f1:
                    alp = alp2
                else:
                    alp = alp1

            if max([i == alp for i in alist]):
                quart = 0
            if quart:
                alp = LSUtils().lsguard(alp, alist, amax, amin, small)
                policy = self.get_policy(x + alp * p, L=stopping_actions)
                avg_metrics = self.eval_theta(policy=policy,
                                              max_steps=self.experiment_config.hparams[
                                                  agents_constants.COMMON.MAX_ENV_STEPS].value)
                falp = round(avg_metrics[env_constants.ENV_METRICS.RETURN], 3)
                alist.append(alp)
                flist.append(falp)
                (alist, flist, abest, fbest, fmed, up, down, monotone,
                 minima, nmin, unitlen, s) = GLSUtils().lssort(alist, flist)
        
        return (alist, flist, amin, amax, alp, abest, fbest, fmed, up, down, monotone,
                minima, nmin, unitlen, s, good, saturated)

    def lssep(self, nloc: int, small: float, sinit: int, short: float, x: NDArray[np.float64], p: NDArray[np.int32],
              alist: List[float], flist: List[float], amin: float, amax: float, alp: float, abest: float, fbest: float,
              fmed: float, up: List[float], down: List[float], monotone: int, minima: List[int], nmin: int,
              unitlen: float, s: int, stopping_actions: int):
        nsep = 0
        while nsep < nmin:
            down = [i < j for i, j in zip(flist[1: s], flist[0: s - 1])]
            sep = [i and j and k for i, j, k in zip([True, True] + down, [False] + up + [False], down + [True, True])]
            temp_sep = [i and j and k for i, j, k in zip([True, True] + up,
                                                         [False] + down + [False], up + [True, True])]
            sep = [i or j for i, j in zip(sep, temp_sep)]

            ind = [i for i in range(len(sep)) if sep[i]]
            
            if len(ind) == 0:
                break
            
            aa = [0.5 * (alist[i] + alist[i - 1]) for i in ind]	# interval midpoints
            if len(aa) > nloc:
                ff = [min(flist[i], flist[j]) for i, j in ind]
                ind = np.argsort(ff)
                ff.sort()
                aa = [aa[ind[i]] for i in range(0, nloc)]

            for alp in aa:
                policy = self.get_policy(x + alp * p, L=stopping_actions)
                avg_metrics = self.eval_theta(policy=policy,
                                              max_steps=self.experiment_config.hparams[
                                                  agents_constants.COMMON.MAX_ENV_STEPS].value)
                falp = round(avg_metrics[env_constants.ENV_METRICS.RETURN], 3)
                alist.append(alp)
                flist.append(falp)
                nsep = nsep + 1
                if nsep >= nmin:
                    break
            (alist, flist, abest, fbest, fmed, up, down, monotone,
             minima, nmin, unitlen, s) = GLSUtils().lssort(alist, flist)

        for times in range(0, nmin - nsep):
            print(times)
            alist, flist, alp, fac = self.lsnew(nloc, small, sinit, short, x, p, s, alist,
                                                flist, amin, amax, alp, abest, fmed, unitlen, stopping_actions)
            (alist, flist, abest, fbest, fmed, up, down, monotone,
             minima, nmin, unitlen, s) = GLSUtils().lssort(alist, flist)

        return (alist, flist, amin, amax, alp, abest, fbest, fmed,
                up, down, monotone, minima, nmin, unitlen, s)

    def lslocal(self, nloc: int, small: float, sinit: int, short: float, x :NDArray[np.float64], p: NDArray[np.int32],
                alist: List[float], flist: List[float],
                amin: float, amax: float, alp: float, abest: float, fbest: float, fmed: float,
                up: List[float], down: List[float], monotone: int,
                minima: List[int], nmin: int, unitlen: float, s: int, saturated: int, stopping_actions: int):
        up = [i < j for i, j in zip(flist[0: s - 1], flist[1: s])]
        down = [i <= j for i, j in zip(flist[1: s], flist[0: s - 1])]
        down[s - 2] = (flist[s - 1] < flist[s - 2])
        minima = [i and j for i, j in zip(up + [True], [True] + down)]
        imin = [i for i in range(len(minima)) if minima[i]]

        ff = [flist[i] for i in imin]
        perm = np.argsort(ff)
        ff.sort()

        imin = [imin[i] for i in perm]
        nind = min(nloc, len(imin))
        imin = imin[nind - 1:: - 1]

        nadd = 0
        nsat = 0
        
        for i in imin:
            if i <= 1:
                ind = [j for j in range(5)]
                ii = i
            elif i >= s - 2:
                ind = [j for j in range(s - 5, s)]
                ii = i - (s - 1) + 4
            else:
                ind = [j for j in range(i - 2, i + 3)]
                ii = 2
            aa = [alist[i] for i in ind]
            ff = [flist[i] for i in ind]
            
            f12 = (ff[1] - ff[0]) / (aa[1] - aa[0])
            f23 = (ff[2] - ff[1]) / (aa[2] - aa[1])
            f34 = (ff[3] - ff[2]) / (aa[3] - aa[2])
            f45 = (ff[4] - ff[3]) / (aa[4] - aa[3])
            f123 = (f23 - f12) / (aa[2] - aa[0])
            f234 = (f34 - f23) / (aa[3] - aa[1])
            f345 = (f45 - f34) / (aa[4] - aa[2])
            if ii == 0:
                cas = 0
                if f123 > 0 and f123 < np.Inf:
                    alp = 0.5 * (aa[1] + aa[2] - f23 / f123)
                    if alp < amin:
                        cas = -1
                else:
                    alp = -np.Inf
                    if alist[0] == amin and flist[1] < flist[2]:
                        cas = -1
                alp = LSUtils().lsguard(alp, alist, amax, amin, small)
            elif ii == 4:
                cas = 0
                if f345 > 0 and f345 < np.Inf:
                    alp = 0.5 * (aa[2] + aa[3] - f34 / f345)
                    if alp > amax:
                        cas = -1
                else:
                    alp = np.Inf
                    if alist[s - 1] == amax and flist[s - 2] < flist[s - 3]:
                        cas = -1
                alp = LSUtils().lsguard(alp, alist, amax, amin, small)
            elif not (f234 > 0 and f234 < np.Inf):
                cas = 0
                if ii < 2:
                    alp = 0.5 * (aa[1] + aa[2] - f23 / f123)
                else:
                    alp = 0.5 * (aa[2] + aa[3] - f34 / f345)

            elif not (f123 > 0 and f123 < np.Inf):
                if f345 > 0 and f345 < np.Inf:
                    cas = 5
                else:

                    cas = 0
                    alp = 0.5 * (aa[2] + aa[3] - f34 / f234)
            elif f345 > 0 and f345 < np.Inf and ff[1] > ff[3]:
                cas = 5
            else:
                cas = 1
            
            if cas == 0:
                alp = max(amin, min(alp, amax))
            elif cas == 1:

                if ff[1] < ff[2]:
                    f13 = (ff[2] - ff[0]) / (aa[2] - aa[0])
                    f1x4 = (f34 - f13) / (aa[3] - aa[0])
                else:
                    f24 = (ff[3] - ff[1]) / (aa[3] - aa[1])
                    f1x4 = (f24 - f12) / (aa[3] - aa[0])
                alp = 0.5 * (aa[1] + aa[2] - f23 / (f123 + f234 - f1x4))
                if alp <= min(aa) or alp >= max(aa):
                    cas = 0
                    alp = 0.5 * (aa[1] + aa[2] - f23 / max(f123, f234))
            elif cas == 5:
                if ff[2] < ff[3]:
                    f24 = (ff[3] - ff[1]) / (aa[3] - aa[1])
                    f2x5 = (f45 - f24) / (aa[4] - aa[1])
                else:
                    f35 = (ff[4] - ff[2]) / (aa[4] - aa[2])
                    f2x5 = (f35 - f23) / (aa[4] - aa[1])
                alp = 0.5 * (aa[2] + aa[3] - f34 / (f234 + f345 - f2x5))
                if alp <= min(aa) or alp >= max(aa):
                    cas = 0
                    alp = 0.5 * (aa[2] + aa[3] - f34 / max(f234, f345))
            if cas < 0 or flist[i] > fmed:
                alptol = 0
            elif cas >= 0:
                if i == 0:
                    alptol = small * (alist[2] - alist[0])
                elif i == s - 1:
                    alptol = small * (alist[s - 1] - alist[s - 3])
                else:
                    alptol = small * (alist[i + 1] - alist[i - 1])
            close = (min([abs(i - alp) for i in alist]) <= alptol)
            if cas < 0 or close:
                nsat = nsat + 1

            saturated = (nsat == nind)
            final = saturated and not max([i == alp for i in alist])
            if cas >= 0 and (final or not close):
                nadd = nadd + 1
                policy = self.get_policy(x + alp * p, L=stopping_actions)
                avg_metrics = self.eval_theta(policy=policy,
                                              max_steps=self.experiment_config.hparams[
                                                  agents_constants.COMMON.MAX_ENV_STEPS].value)
                falp = round(avg_metrics[env_constants.ENV_METRICS.RETURN], 3)
                alist.append(alp)
                flist.append(falp)
        if nadd:
            (alist, flist, abest, fbest, fmed, up, down, monotone,
             minima, nmin, unitlen, s) = GLSUtils().lssort(alist, flist)
        return (alist, flist, alp, abest, fbest, fmed, up, down, monotone,
                minima, nmin, unitlen, s, saturated)
