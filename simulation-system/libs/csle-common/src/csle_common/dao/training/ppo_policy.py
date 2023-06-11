from typing import List, Dict, Union
import numpy as np
import torch
import math
import iteround
from stable_baselines3 import PPO
from csle_common.dao.training.policy import Policy
from csle_common.dao.training.agent_type import AgentType
from csle_common.dao.training.player_type import PlayerType
from csle_common.dao.simulation_config.state import State
from csle_common.dao.simulation_config.action import Action
from csle_common.dao.training.experiment_config import ExperimentConfig
from csle_common.dao.training.policy_type import PolicyType
from csle_common.logging.log import Logger


class PPOPolicy(Policy):
    """
    A neural network policy learned with PPO
    """

    def __init__(self, model, simulation_name: str, save_path: str, player_type: PlayerType, states: List[State],
                 actions: List[Action], experiment_config: ExperimentConfig, avg_R: float):
        """
        Initializes the policy

        :param model: the PPO model
        :param simulation_name: the simulation name
        :param save_path: the path to save the model to
        :param states: list of states (required for computing stage policies)
        :param actions: list of actions
        :param experiment_config: the experiment configuration for training the policy
        :param avg_R: the average reward of the policy when evaluated in the simulation
        """
        super(PPOPolicy, self).__init__(agent_type=AgentType.PPO, player_type=player_type)
        self.model = model
        self.id = -1
        self.simulation_name = simulation_name
        self.save_path = save_path
        if self.model is None:
            try:
                self.model = PPO.load(path=self.save_path)
            except Exception as e:
                Logger.__call__().get_logger().warning(
                    f"There was an exception loading the model from path: {self.save_path}, "
                    f"exception: {str(e)}, {repr(e)}")
        self.states = states
        self.actions = actions
        self.experiment_config = experiment_config
        self.avg_R = avg_R
        self.policy_type = PolicyType.PPO

    def action(self, o: List[float]) -> Union[int, List[int], np.ndarray]:
        """
        Multi-threshold stopping policy

        :param o: the current observation
        :return: the selected action
        """
        a, _ = self.model.predict(np.array(o), deterministic=False)
        return a

    def probability(self, o: List[float], a: int) -> float:
        """
        Multi-threshold stopping policy

        :param o: the current observation
        :param o: the action
        :return: the probability of the action
        """
        prob = math.exp(self.model.policy.get_distribution(obs=torch.tensor([o]).to(self.model.device)).log_prob(
            actions=torch.tensor(a)).item())
        return prob

    def to_dict(self) -> Dict[str, Union[float, int, str]]:
        """
        :return: a dict representation of the policy
        """
        d = {}
        d["id"] = self.id
        d["simulation_name"] = self.simulation_name
        d["save_path"] = self.save_path
        if self.model is not None:
            d["policy_kwargs"] = self.model.policy_kwargs
            self.model.save(path=self.save_path)
        else:
            d["policy_kwargs"] = {}
            d["policy_kwargs"]["net_arch"] = []
        d["states"] = list(map(lambda x: x.to_dict(), self.states))
        d["player_type"] = self.player_type
        d["actions"] = list(map(lambda x: x.to_dict(), self.actions))
        d["experiment_config"] = self.experiment_config.to_dict()
        d["agent_type"] = self.agent_type
        d["avg_R"] = self.avg_R
        d["policy_type"] = self.policy_type
        return d

    @staticmethod
    def from_dict(d: Dict) -> "PPOPolicy":
        """
        Converst a dict representation of the object to an instance

        :param d: the dict to convert
        :return: the created instance
        """
        obj = PPOPolicy(model=None, simulation_name=d["simulation_name"], save_path=d["save_path"],
                        states=list(map(lambda x: State.from_dict(x), d["states"])), player_type=d["player_type"],
                        actions=list(map(lambda x: Action.from_dict(x), d["actions"])),
                        experiment_config=ExperimentConfig.from_dict(d["experiment_config"]), avg_R=d["avg_R"])
        obj.id = d["id"]
        return obj

    def stage_policy(self, o: Union[List[Union[int, float]], int, float]) -> List[List[float]]:
        """
        Gets the stage policy, i.e a |S|x|A| policy

        :param o: the latest observation
        :return: the |S|x|A| stage policy
        """
        stage_strategy = np.zeros((len(self.states), len(self.actions)))
        for i, s_a in enumerate(self.states):
            o[0] = s_a
            for j, a in enumerate(self.actions):
                stage_strategy[i][j] = self.probability(o=o, a=j)
            stage_strategy[i] = iteround.saferound(stage_strategy[i], 2)
            assert round(sum(stage_strategy[i]), 3) == 1
        return stage_strategy.tolist()

    def _get_attacker_dist(self, obs) -> List[float]:
        """
        Utility function for getting the action distribution conditioned on a given observation

        :param obs: the observation to condition on
        :return: the conditional ation distribution
        """
        obs = np.array([obs])
        actions, values, log_prob = self.model.policy.forward(obs=torch.tensor(obs).to(self.model.device))
        action = actions[0]
        if action == 1:
            stop_prob = math.exp(log_prob)
        else:
            stop_prob = 1 - math.exp(log_prob)
        return [1 - stop_prob, stop_prob]

    def __str__(self) -> str:
        """
        :return: a string representation of the object
        """
        return f"model: {self.model}, id: {self.id}, simulation_name: {self.simulation_name}, " \
               f"save path: {self.save_path}, states: {self.states}, experiment_config: {self.experiment_config}," \
               f"avg_R: {self.avg_R}, policy type: {self.policy_type}"

    @staticmethod
    def from_json_file(json_file_path: str) -> "PPOPolicy":
        """
        Reads a json file and converts it to a DTO

        :param json_file_path: the json file path
        :return: the converted DTO
        """
        import io
        import json
        with io.open(json_file_path, 'r') as f:
            json_str = f.read()
        return PPOPolicy.from_dict(json.loads(json_str))

    def copy(self) -> "PPOPolicy":
        """
        :return: a copy of the DTO
        """
        return self.from_dict(self.to_dict())
