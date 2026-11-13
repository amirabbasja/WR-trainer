import torch


def resolve_cuda_devices(selection, *, unavailable):
    """Resolve validated configuration to explicit, visible CUDA devices."""
    if not torch.cuda.is_available():
        raise ValueError(unavailable)
    count = torch.cuda.device_count()
    if isinstance(selection, list):
        indices = selection
    else:
        device = torch.device(selection)
        if device.type != 'cuda':
            raise ValueError(unavailable)
        indices = [torch.cuda.current_device() if device.index is None else device.index]
    for index in indices:
        if not 0 <= index < count:
            raise ValueError(f'device selects unavailable cuda:{index}; {count} CUDA device(s) visible')
    return tuple(torch.device('cuda', index) for index in indices)
