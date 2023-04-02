import torch


def normalize_samples(x, eps=1e-6):
    x = x - x.mean(dim=0)
    std_a = torch.sqrt((x ** 2).sum(dim=0))
    std_non_zero = std_a > eps
    x = torch.where(std_non_zero, x / std_a, x)

    return x


def process_column(x, task_type='regression'):
    if task_type == 'classification':
        return x.factorize()
    return (x - x.mean()) / x.std(), [None]
