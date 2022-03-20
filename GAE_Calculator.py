import torch
from Device import DEVICE


def calculate_gaes(next_value: torch.tensor,
                   rewards: list,
                   values: list,
                   masks: list,
                   gamma: float = 0.99,
                   lambda_: float = 0.95):
    """
    Return the General Advantage Estimates from the given rewards and values.
    Paper: https://arxiv.org/pdf/1506.02438.pdf
    """
    values: list = values + [next_value]
    gae: torch.tensor = torch.zeros(1, device=DEVICE).unsqueeze(0)
    returns: list = []
    for step in reversed(range(len(rewards))):
        delta: torch.tensor = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * lambda_ * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns
