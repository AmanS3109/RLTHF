import torch

def pairwise_accuracy(reward_chosen, reward_rejected):
    """
    Compute the % of times reward(chosen) > reward(rejected)
    """
    return (reward_chosen > reward_rejected).float().mean().item()