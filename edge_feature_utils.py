import torch

BASE_EDGE_FEATURE_COUNT = 4
PAYMENT_FORMAT_INDEX = 3


def average_residual_update(current, update):
    return (current + update) / 2


def get_edge_type_index():
    return PAYMENT_FORMAT_INDEX


def get_non_normalized_edge_feature_indices(model_name):
    return (PAYMENT_FORMAT_INDEX,) if model_name == "rgcn" else ()


def get_port_feature_indices(use_ports):
    if not use_ports:
        return ()

    return tuple(range(BASE_EDGE_FEATURE_COUNT, BASE_EDGE_FEATURE_COUNT + 2))


def get_time_delta_feature_indices(use_ports, use_tds):
    if not use_tds:
        return ()

    start_idx = BASE_EDGE_FEATURE_COUNT + (2 if use_ports else 0)
    return tuple(range(start_idx, start_idx + 2))


def z_norm(data):
    std = data.std(0).unsqueeze(0)
    std = torch.where(std == 0, torch.ones_like(std), std)
    return (data - data.mean(0).unsqueeze(0)) / std


def z_norm_except(data, excluded_indices):
    if not excluded_indices:
        return z_norm(data)

    normalized = data.clone()
    excluded_indices = set(excluded_indices)
    include_indices = [idx for idx in range(data.shape[1]) if idx not in excluded_indices]
    if include_indices:
        normalized[:, include_indices] = z_norm(data[:, include_indices])

    return normalized
