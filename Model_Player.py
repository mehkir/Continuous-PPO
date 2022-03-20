import gym
import torch
from Model import ActorCriticNetwork
from Device import DEVICE
import numpy as np

MODEL_FILENAME: str = ''
ENV_ID: str = 'Ant-v2'

if __name__ == '__main__':
    HIDDEN_SIZE: int = 64
    env = gym.make(ENV_ID)
    model: ActorCriticNetwork = ActorCriticNetwork(env.observation_space.shape[0], env.action_space.shape[0], HIDDEN_SIZE)
    model = model.to(DEVICE)
    model.load_state_dict(torch.load('./checkpoints/'+ MODEL_FILENAME))
    observation: np.ndarray = env.reset()
    done: bool = False
    total_reward: int = 0
    while not done:
        observation: torch.tensor = torch.tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        distribution, _ = model(observation)
        action: np.ndarray = distribution.mean.detach().cpu().numpy()[0]
        next_observation, reward, done, _ = env.step(action)
        env.render()
        observation: np.ndarray = next_observation
        total_reward += reward
    print('Total reward', total_reward)
    env.close()
