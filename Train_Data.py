from enum import Enum


class TrainData(Enum):
    """Used for index train data"""
    OBSERVATIONS = 0
    ACTIONS = 1
    REWARDS = 2
    VALUES = 3
    ACTION_LOG_PROBABILITIES = 4
    MASKS = 5
    ADVANTAGES = 6
    RETURNS = 7
