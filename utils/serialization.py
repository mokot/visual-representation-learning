import torch
from pathlib import Path


def custom_serializer(obj: object) -> object:
    """
    Custom serializer for handling non-serializable objects.

    Args:
        obj: The object to serialize.

    Returns:
        A JSON-serializable representation of the object.

    Raises:
        TypeError: If the object type is unsupported.
    """
    if isinstance(obj, torch.Tensor):
        return obj.tolist()  # Convert tensors to lists
    elif isinstance(obj, Path):
        return str(obj)  # Convert Path to string
    elif hasattr(obj, "__dict__"):  # Handle objects with attributes
        return obj.__dict__
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
