import torch


def calculate_l1_error(x: torch.Tensor, full_x: torch.Tensor) -> float:
    return torch.abs(x - full_x).mean().item()


def calculate_l2_error(x: torch.Tensor, full_x: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((x - full_x) ** 2)).item()


def calculate_relative_l1_error(x: torch.Tensor, full_x: torch.Tensor, eps: float = 1e-10) -> float:
    error = torch.abs(x - full_x) / (torch.abs(full_x) + eps)
    return error.mean().item()


def calculate_relative_l2_error(x: torch.Tensor, full_x: torch.Tensor, eps: float = 1e-10) -> float:
    error = torch.abs(x - full_x) / (torch.abs(full_x) + eps)
    return torch.sqrt(torch.mean(error ** 2)).item()


def calculate_cosine_similarity_error(x: torch.Tensor, full_x: torch.Tensor) -> float:
    x_flat = x.view(x.size(0), -1)
    full_x_flat = full_x.view(full_x.size(0), -1)
    cosine_sim = torch.nn.functional.cosine_similarity(x_flat, full_x_flat, dim=1)
    return (1 - cosine_sim.mean()).item()


def calculate_all_errors(x: torch.Tensor, full_x: torch.Tensor, eps: float = 1e-10) -> dict:
    return {
        'l1': calculate_l1_error(x, full_x),
        'l2': calculate_l2_error(x, full_x),
        'relative_l1': calculate_relative_l1_error(x, full_x, eps),
        'relative_l2': calculate_relative_l2_error(x, full_x, eps),
        'cosine_similarity': calculate_cosine_similarity_error(x, full_x),
    }

