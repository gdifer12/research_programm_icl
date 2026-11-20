"""Explicit execution backends; unavailable accelerators never fall back to CPU."""
import os
import torch


def require_device(device, precision=None):
    device = torch.device(device)
    if device.type not in {'cpu', 'cuda', 'mps'}:
        raise ValueError('Supported devices: mps, cpu and cuda')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; CPU fallback is disabled')
    if device.type == 'mps':
        if device.index not in {None, 0}:
            raise ValueError('MPS supports one device: use mps or mps:0')
        if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
            raise RuntimeError('MPS requested but unavailable; use native macOS with Metal access. CPU fallback is disabled')
        if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') != '0':
            raise ValueError('Set PYTORCH_ENABLE_MPS_FALLBACK=0 before starting Python; implicit CPU fallback is not allowed')
        if precision not in {None, 'fp32'}:
            raise ValueError('The validated MPS pipeline requires fp32')
    elif precision == 'fp16' and device.type != 'cuda':
        raise ValueError('fp16 training requires CUDA')
    return device


def synchronize(device):
    device = torch.device(device)
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize(device)


def memory_stats(device):
    device = torch.device(device)
    if device.type == 'mps':
        # MPS exposes current allocations, not a resettable peak-memory counter.
        return {'allocated_bytes': torch.mps.current_allocated_memory(),
                'driver_allocated_bytes': torch.mps.driver_allocated_memory(),
                'recommended_max_memory_bytes': torch.mps.recommended_max_memory()}
    if device.type == 'cuda':
        return {'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)}
    return {}


def device_info(device):
    device = torch.device(device)
    result = {'mps_built': torch.backends.mps.is_built(),
              'mps_available': torch.backends.mps.is_available()}
    if device.type == 'mps':
        result.update(gpu='Apple Metal GPU', mps_cpu_fallback=False,
                      memory_counter_scope='current allocations, not peak', **memory_stats(device))
    return result
