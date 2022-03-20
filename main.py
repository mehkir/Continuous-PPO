import gym
import torch
import numpy as np
from Model import ActorCriticNetwork
from Device import DEVICE
from PPO_Trainer import PPOTrainer
from Data_Sampler import DataSampler
from Train_Data import TrainData
from datetime import date
from datetime import datetime

ENV_ID: str = 'Ant-v2'
TARGET_REWARD: int = 2500
NUMBER_OF_TESTS: int = 10
HIDDEN_SIZE: int = 64
SAMPLE_STEPS: int = 2048
TEST_EPISODE_FREQUENCY: int = 20
MINI_BATCH_SIZE: int = 64
RESUME_MODEL: bool = False
MODEL_FILENAME: str = ''
LOG_FILE_CREATION_DATE: str = f'{date.today().strftime("%b-%d-%Y")} {datetime.now().strftime("%H:%M:%S")}'
NAME_OF_RUN: str = 'Basic training'
LOG_FILE_NAME: str = f'{NAME_OF_RUN} {LOG_FILE_CREATION_DATE}'

if __name__ == '__main__':
    # Initialize log
    with open(f'./logs/{LOG_FILE_NAME}.csv', 'a', encoding='utf-8') as file:
        file.write('Episode,Average Reward\n')
    # Initialize train environment
    train_env = gym.make(ENV_ID)
    # Initialize test environment
    test_env = gym.make(ENV_ID)
    # Initialize model
    model: ActorCriticNetwork = ActorCriticNetwork(train_env.observation_space.shape[0], train_env.action_space.shape[0], HIDDEN_SIZE)
    model = model.to(DEVICE)
    # Initialize data sampler
    data_sampler: DataSampler = DataSampler()
    # Initialize PPO Trainer
    ppo: PPOTrainer = PPOTrainer(
        model,
        policy_lr=1e-4,
        max_policy_train_iters=10,
        mini_batch_size=MINI_BATCH_SIZE
    )
    train_observation: np.ndarray = train_env.reset()
    best_reward: float = None
    episode: int = 0
    # Resume model
    if RESUME_MODEL:
        model.load_state_dict(torch.load('./checkpoints/' + MODEL_FILENAME))
        # Reward shown in the model name
        best_reward = 1234.1234
        # Episode shown in the model name
        episode = 1234
        # For correct tracking of the episode on resume, it must be decreased by 1
        # because it is stored incremented by 1 in the model name.
        episode = episode-1
    early_stop: bool = False
    # Training loop
    while not early_stop:
        # Perform rollout
        train_data, train_observation = data_sampler.sample_data(model, train_env, train_observation, sample_steps=SAMPLE_STEPS)
        # Retain all values from the list and put them concatenated in a torch tensor
        train_data[TrainData.OBSERVATIONS.value] = torch.cat(train_data[TrainData.OBSERVATIONS.value]).unsqueeze(1)
        train_data[TrainData.ACTIONS.value] = torch.cat(train_data[TrainData.ACTIONS.value]).unsqueeze(1)
        train_data[TrainData.ACTION_LOG_PROBABILITIES.value] = torch.cat(train_data[TrainData.ACTION_LOG_PROBABILITIES.value]).detach().unsqueeze(1)
        train_data[TrainData.RETURNS.value] = torch.cat(train_data[TrainData.RETURNS.value]).detach().unsqueeze(1)
        train_data[TrainData.VALUES.value] = torch.cat(train_data[TrainData.VALUES.value]).detach().unsqueeze(1)
        train_data[TrainData.REWARDS.value] = torch.cat(train_data[TrainData.REWARDS.value])

        # Calculate advantages
        train_data[TrainData.ADVANTAGES.value] = [return_ - value for return_, value in zip(train_data[TrainData.RETURNS.value], train_data[TrainData.VALUES.value])]
        train_data[TrainData.ADVANTAGES.value] = torch.cat(train_data[TrainData.ADVANTAGES.value]).unsqueeze(1)
        train_data[TrainData.ADVANTAGES.value] = (train_data[TrainData.ADVANTAGES.value] - train_data[TrainData.ADVANTAGES.value].mean()) / (train_data[TrainData.ADVANTAGES.value].std() + 1e-8)

        # Train model
        ppo.train_model(train_data[TrainData.OBSERVATIONS.value],
                        train_data[TrainData.ACTIONS.value],
                        train_data[TrainData.ACTION_LOG_PROBABILITIES.value],
                        train_data[TrainData.RETURNS.value],
                        train_data[TrainData.ADVANTAGES.value])
        episode = episode + 1
        if (episode + 1) % TEST_EPISODE_FREQUENCY == 0:
            total_rewards = np.empty(NUMBER_OF_TESTS)
            for test_idx in range(NUMBER_OF_TESTS):
                test_observation: np.ndarray = test_env.reset()
                done: bool = False
                total_reward: float = 0.0
                while not done:
                    test_observation: torch.tensor = torch.tensor(test_observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    distribution, _ = model(test_observation)
                    action: np.ndarray = distribution.mean.detach().cpu().numpy()[0]
                    next_observation, reward, done, _ = test_env.step(action)
                    test_observation: np.ndarray = next_observation
                    total_reward += reward
                total_rewards[test_idx] = total_reward
            print('Episode {} | Avg Reward {:.1f}'.format(
                episode + 1, np.mean(total_rewards)
            ))
            with open(f'./logs/{LOG_FILE_NAME}.csv', 'a', encoding='utf-8') as file:
                file.write(f'{episode + 1},{np.mean(total_rewards)}\n')
            if best_reward is None or np.mean(total_rewards) > best_reward:
                best_reward = np.mean(total_rewards)
                filename: str = '{}-model {} reward={} episodes={}.pth'.format(ENV_ID,NAME_OF_RUN,best_reward,episode + 1)
                checkpoint_path: str = './checkpoints/'
                torch.save(model.state_dict(), checkpoint_path+filename)
                print('Saved checkpoint: {}'.format(filename))
            if np.mean(total_rewards) > TARGET_REWARD: early_stop = True