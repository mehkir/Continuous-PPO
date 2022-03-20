import numpy as np
import torch
from torch import optim
from Model import ActorCriticNetwork
CRITIC_COEFFICIENT = 0.5
ENTROPY_COEFFICIENT = 0.001


class PPOTrainer:
    """
    Trains the given model with Proximate Policy Optimization
    Paper: https://arxiv.org/pdf/1707.06347.pdf
    """
    def __init__(self, model: ActorCriticNetwork,
                 ppo_clip_val: float = 0.2,
                 max_policy_train_iters: int = 10,
                 policy_lr: float = 1e-4,
                 mini_batch_size: int =64):
        self.model: ActorCriticNetwork = model
        self.ppo_clip_val: float = ppo_clip_val
        self.max_policy_train_iters: int = max_policy_train_iters
        self.mini_batch_size: int = mini_batch_size
        self.optimizer:optim.adam.Adam = optim.Adam(self.model.parameters(), lr=policy_lr)

    def ppo_iter(self,
                 observations: torch.tensor,
                 actions: torch.tensor,
                 action_log_probs: torch.tensor,
                 returns: torch.tensor,
                 advantages: torch.tensor):
        batch_size: int = len(observations)
        # generates random mini-batches until we have covered the full batch
        for _ in range(batch_size // self.mini_batch_size):
            indices: np.ndarray = np.random.randint(0, batch_size, self.mini_batch_size)
            yield observations[indices], \
                  actions[indices],\
                  action_log_probs[indices],\
                  returns[indices],\
                  advantages[indices]

    def train_model(self,
                    observations: torch.tensor,
                    actions: torch.tensor,
                    old_action_log_probs: torch.tensor,
                    returns: torch.tensor,
                    advantages: torch.tensor):
        for _ in range(self.max_policy_train_iters):
            for observation_batch, action_batch, old_action_log_probs_batch, return_batch, advantage_batch in \
                                                                      self.ppo_iter(observations,
                                                                                    actions,
                                                                                    old_action_log_probs,
                                                                                    returns,
                                                                                    advantages):
                distribution, value = self.model(observation_batch)
                entropy: torch.tensor = distribution.entropy().mean()
                new_action_log_probs: torch.tensor = distribution.log_prob(action_batch)
                # the ratio part of L^(CPI) exponentiated, since the probabilities are log-probabilities
                policy_ratio: torch.tensor = torch.exp(new_action_log_probs - old_action_log_probs_batch)
                trpo_term: torch.tensor = policy_ratio * advantage_batch # full loss
                # clip(r_t(theta), 1-epsilon, 1+epsilon)
                clipped_ratio: torch.tensor = torch.clamp(policy_ratio, 1.0 - self.ppo_clip_val, 1.0 + self.ppo_clip_val)
                clipped_term: torch.tensor = clipped_ratio * advantage_batch # clipped loss
                # min(full loss,clipped loss)
                actor_loss: torch.tensor = torch.min(trpo_term, clipped_term).mean()
                critic_loss: torch.tensor = (return_batch - value).pow(2).mean()
                # The loss from https://arxiv.org/pdf/1707.06347.pdf is: L_{t}^{CLIP+VF+S}(\theta)
                # Since stochastic gradient has to be applied, a minus sign before the loss is necessary.
                loss: torch.tensor = -(actor_loss - CRITIC_COEFFICIENT * critic_loss + ENTROPY_COEFFICIENT * entropy)
                self.optimizer.zero_grad()
                loss.backward()  # calculate gradients
                self.optimizer.step()  # Optimize gradient according to derivatives from backpropagation