import gym
import torch
import numpy as np
from torch.distributions.normal import Normal
from Model import ActorNetwork
from Device import DEVICE

MODEL_FILENAME='Ant-v2-Actormodel reward=3369.0882817377146 episodes=1900.pth'
ENV_ID = 'Ant-v2'

if __name__ == '__main__':
    HIDDEN_SIZE = 64
    env = gym.make(ENV_ID)
    actor_model: ActorNetwork = ActorNetwork(env.observation_space.shape[0], env.action_space.shape[0], HIDDEN_SIZE)
    actor_model = actor_model.to(DEVICE)
    actor_model.load_state_dict(torch.load('./checkpoints/actor lr 1e-4 critic 1e-3/'+ MODEL_FILENAME))
    observation: np.ndarray = env.reset()
    done: bool = False
    total_reward: float = 0.0
    while not done:
        observation: torch.tensor = torch.tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        distribution: Normal = actor_model(observation)
        action: np.ndarray = distribution.mean.detach().cpu().numpy()[0]
        next_observation, reward, done, _ = env.step(action)
        env.render()
        observation: np.ndarray = next_observation
        total_reward += reward
    print('Total reward', total_reward)
    env.close()
