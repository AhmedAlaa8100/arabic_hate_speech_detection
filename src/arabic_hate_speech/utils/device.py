"""
Device utilities for Arabic Hate Speech Detection.
"""

import torch
from typing import Union


def get_device(device: str = "auto", force_cpu: bool = False) -> torch.device:
    """
    Get the appropriate device for computation.
    
    Args:
        device: Device specification ("auto", "cuda", "cpu")
        force_cpu: Force CPU usage even if CUDA is available
        
    Returns:
        PyTorch device object
    """
    if force_cpu:
        return torch.device("cpu")
    
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    return torch.device(device)


def get_device_info() -> dict:
    """
    Get information about available devices.
    
    Returns:
        Dictionary containing device information
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": str(get_device()),
    }
    
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_memory_allocated"] = torch.cuda.memory_allocated(0)
        info["cuda_memory_reserved"] = torch.cuda.memory_reserved(0)
    
    return info
