import torch
from torch import nn
from torch.distributions.normal import Normal


class ActorCriticNetwork(nn.Module):
    """The neural net with actor and critic layers"""
    def __init__(self,
                 obs_space_size: int,
                 action_space_size: int,
                 hidden_size: int,
                 std: float = 0.0):
        super(ActorCriticNetwork, self).__init__()

        # Actor layers
        self.actor_layers = nn.Sequential(
            nn.Linear(obs_space_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_space_size)
        )

        # Critic layers
        self.critic_layers = nn.Sequential(
            nn.Linear(obs_space_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.log_std = nn.Parameter(torch.ones(1, action_space_size) * std)

    def forward_to_critic(self, observation: torch.tensor) -> torch.tensor:
        """Forwards the observation to the critic layers and returns a value"""
        return self.critic_layers(observation)

    def forward_to_actor(self, observation: torch.tensor) -> Normal:
        """Forwards the observation to the actor layers and returns a normal distribution over the action"""
        policy_logits = self.actor_layers(observation)
        std = self.log_std.exp().expand_as(policy_logits)
        dist = Normal(policy_logits, std)
        return dist

    def forward(self, observation: torch.tensor) -> (Normal, torch.tensor):
        """Forwards the observation to the actor and critic layers, and returns their outputs"""
        dist = self.forward_to_actor(observation)
        value = self.forward_to_critic(observation)
        return dist, value
