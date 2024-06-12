from abc import ABC, abstractmethod
from typing import List, Any
from csle_common.dao.simulation_config.base_env import BaseEnv
from csle_common.dao.simulation_config.simulation_trace import SimulationTrace


class EnvPoisoning(ABC):
    @abstractmethod
    def attack_callback(self, observation: List[float], reward: float, done: bool, info: dict, iteration: int) -> List[float]:
        pass

    def env_wrapper(self, env: BaseEnv):
        return PoisonedEnvWrapper(env, self)

class PoisonedEnvWrapper(BaseEnv):
    def __init__(self, env: BaseEnv, poisoning_strategy: EnvPoisoning):
        super().__init__()
        self.env = env
        self.poisoning_strategy = poisoning_strategy
        self.iteration = 0

        # Copying the attributes from the original environment
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.reward_range = env.reward_range
        self.metadata = env.metadata

    def step(self, action, attack_mode=True):
        observation, reward, done, _, info = self.env.step(action)
        if attack_mode:
            # Apply the poisoning strategy to the env step returns
            observation, reward, done, _, info = self.poisoning_strategy.attack_callback(observation, reward, done, done, info, self.iteration)
        self.iteration += 1
        return observation, reward, done, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        # Apply the poisoning strategy to the observation
        poisoned_observation = self.poisoning_strategy.attack_callback(observation, 0, self.iteration)
        self.iteration = 0
        return poisoned_observation

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

    def __getattr__(self, name):
        return getattr(self.env, name)

    # Implementing BaseEnv abstract methods
    def get_traces(self) -> List[SimulationTrace]:
        return self.env.get_traces()

    def reset_traces(self) -> None:
        self.traces = []

    def manual_play(self) -> None:
        self.env.manual_play()

    def set_model(self, model: Any) -> None:
        self.env.set_model(model)

    def set_state(self, state: Any) -> None:
        self.env.set_state(state)
