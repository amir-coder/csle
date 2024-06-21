from abc import ABC, abstractmethod
from typing import List, Any
from csle_common.dao.simulation_config.base_env import BaseEnv
from csle_common.dao.simulation_config.simulation_trace import SimulationTrace
from gym_csle_stopping_game.util.stopping_game_util import StoppingGameUtil
from collections import deque
import json

class IntrusionLengthTracker:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.intrusion_lengths = deque(maxlen=window_size)

    def add_intrusion_length(self, length):
        self.intrusion_lengths.append(length)
    
    def add_intrusion_length_vec(self, lengths):
        self.intrusion_length.extend(lengths)

    def get_average_intrusion_length(self):
        return sum(self.intrusion_lengths) / len(self.intrusion_lengths)
    
    def intrusion_length_absolute_progress(self):
        return abs(self.intrusion_lengths[-1] - self.intrusion_lengths[0])
    
    def smooth_intrusion_length_absolute_progress(self, smoothing_window=10):
        return abs(sum(list(self.intrusion_lengths)[-smoothing_window:])/smoothing_window - sum(list(self.intrusion_lengths)[:smoothing_window])/smoothing_window)



class MetaAgent:
    def __init__(self, initial_reward_vector, learning_rate=0.1, intrusion_length_tracker=None, reward_vector_update_freq=20):
        self.reward_vector = initial_reward_vector
        self.learning_rate = learning_rate
        self.intrusion_length_tracker = intrusion_length_tracker or IntrusionLengthTracker()
        self.reward_vector_update_freq = reward_vector_update_freq
        self.reward_vector_record = deque([list(initial_reward_vector)])  # Using deque for the record

    def update_reward_vector(self):
        """
        Update the reward vector based on the intrusion length change.
        """
        intrusion_length_change = self.intrusion_length_tracker.smooth_intrusion_length_absolute_progress()
        for i in range(len(self.reward_vector)):
            if self.reward_vector[i] < 0:
                self.reward_vector[i] -= self.learning_rate * intrusion_length_change
            else:
                self.reward_vector[i] += self.learning_rate * intrusion_length_change

        self.reward_vector_record.append(list(self.reward_vector))  # Record the updated reward vector
        return self.reward_vector

    def get_recent_rewards(self, n):
        """
        Get the most recent n reward vectors.
        """
        return list(self.reward_vector_record)[-n:]  # Convert to list for slicing

    def record_reward_to_file(self, filename):
        """
        Save the recorded reward vectors to a JSON file.
        
        :param filename: The file to save the reward vectors.
        """
        with open(filename, 'w') as file:
            json.dump(list(self.reward_vector_record), file, indent=4)


class EnvPoisoning(ABC):
    @abstractmethod
    def attack_callback(self, observation: List[float], reward: float, done: bool, done_: bool, info: dict, iteration: int):
                                pass

    def env_wrapper(self, env: BaseEnv, meta_agent: MetaAgent):
        return PoisonedEnvWrapper(env, self, meta_agent)

class PoisonedEnvWrapper(BaseEnv):
    def __init__(self, env: BaseEnv, poisoning_strategy: EnvPoisoning, meta_agent: MetaAgent):
        super().__init__()
        self.env = env
        self.poisoning_strategy = poisoning_strategy
        self.iteration = 0
        self.meta_agent = meta_agent

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

        if done:
             self.meta_agent.intrusion_length_tracker.add_intrusion_length(length=info['intrusion_end'])

        if self.iteration +1 % self.meta_agent.reward_vector_update_freq == 0:
             reward_tensor = self.meta_agent.update_reward_vector()
             self.config.simulation_env_input_config.stopping_game_config.R = list(StoppingGameUtil.reward_tensor(R_INT=reward_tensor[0], R_COST=reward_tensor[1], R_SLA=reward_tensor[2], R_ST=reward_tensor[3], L=3))
        
        return observation, reward, done, done, info

    def reset(self, **kwargs):
        self.iteration = 0
        return self.env.reset(**kwargs)

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
