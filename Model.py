import torch
from torch import nn
from torch.distributions.normal import Normal


class ActorNetwork(nn.Module):
    """The neural net with actor and critic layers"""
    def __init__(self,
                 obs_space_size: int,
                 action_space_size: int,
                 hidden_size: int,
                 std: float = 0.0):
        super(ActorNetwork, self).__init__()

        # Actor layers
        self.policy_layers = nn.Sequential(
            nn.Linear(obs_space_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_space_size)
        )
        self.log_std: torch.tensor = nn.Parameter(torch.ones(1, action_space_size) * std)

    def forward(self, observation) -> Normal:
        """Forwards the observation to the actor layers and returns a normal distribution over the action"""
        policy_logits: torch.tensor = self.policy_layers(observation)
        std: torch.tensor = self.log_std.exp().expand_as(policy_logits)
        dist: Normal = Normal(policy_logits, std)
        return dist


class CriticNetwork(nn.Module):
    """The neural net with actor and critic layers"""
    def __init__(self, obs_space_size, hidden_size):
        super(CriticNetwork, self).__init__()

        # Critic layers
        self.value_layers = nn.Sequential(
            nn.Linear(obs_space_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, observation) -> torch.tensor:
        """Forwards the observation to the critic layers and returns a value"""
        value: torch.tensor = self.value_layers(observation)
        return value