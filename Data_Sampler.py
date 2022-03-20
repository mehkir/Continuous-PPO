import torch
import numpy as np
import GAE_Calculator
from Device import DEVICE
from Train_Data import TrainData
from Model import ActorCriticNetwork


class DataSampler:

    def sample_data(self,
                    model: ActorCriticNetwork,
                    environment,
                    observation: np.ndarray,
                    sample_steps: int = 2048):
        """Samples data and/or trajectories"""
        train_data: list = list()
        for _ in range(len(TrainData)):
            train_data.append(list())
        for _ in range(sample_steps):
            observation: torch.tensor = torch.tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            distribution, value = model(observation)
            action: torch.tensor = distribution.sample()
            next_observation, reward, done, _ = environment.step(action.cpu().numpy()[0])
            action_log_prob: torch.tensor = distribution.log_prob(action)
            reward: torch.tensor = torch.tensor(reward, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            reward = reward.unsqueeze(0)
            mask: torch.tensor = torch.tensor(1 - done, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            mask = mask.unsqueeze(0)

            train_data[TrainData.OBSERVATIONS.value].append(observation)
            train_data[TrainData.ACTIONS.value].append(action)
            train_data[TrainData.REWARDS.value].append(reward)
            train_data[TrainData.VALUES.value].append(value)
            train_data[TrainData.ACTION_LOG_PROBABILITIES.value].append(action_log_prob)
            train_data[TrainData.MASKS.value].append(mask)

            if done:
                observation: np.ndarray = environment.reset()
            else:
                observation: np.ndarray = next_observation

        next_observation: torch.tensor = torch.tensor(next_observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        _, next_value = model(next_observation)
        train_data[TrainData.RETURNS.value]: list = GAE_Calculator.calculate_gaes(next_value, train_data[TrainData.REWARDS.value],
                                                             train_data[TrainData.VALUES.value],
                                                             train_data[TrainData.MASKS.value])
        return train_data, observation
