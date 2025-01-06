import h5py
import torch
from typing import List
from pathlib import Path


def load_splat(path: Path, device: torch.device) -> torch.nn.ParameterDict:
    """
    Load a single splat (ParameterDict) from a Torch file.

    Args:
        path (Path): The file path to load the splat.
        device (torch.device): The device to load the splat onto.

    Returns:
        torch.nn.ParameterDict: A ParameterDict object representing the splat.
    """
    splat_dict = torch.load(path)
    return torch.nn.ParameterDict(
        {key: torch.nn.Parameter(value.to(device)) for key, value in splat_dict.items()}
    )


def load_splat_hdf5(path: Path, device: torch.device) -> torch.nn.ParameterDict:
    """
    Load a single splat (ParameterDict) from an HDF5 file.

    Args:
        path (Path): The file path to load the splat.
        device (torch.device): The device to load the splat onto.

    Returns:
        torch.nn.ParameterDict: A ParameterDict object representing the splat.
    """
    with h5py.File(path, "r") as f:
        return torch.nn.ParameterDict(
            {
                key: torch.nn.Parameter(torch.tensor(f[key][:], device=device))
                for key in f.keys()
            }
        )


def load_splats(path: Path, device: torch.device) -> List[torch.nn.ParameterDict]:
    """
    Load a list of splats (ParameterDict) from a Torch file.

    Args:
        path (Path): The file path to load the splats.
        device (torch.device): The device to load the splats onto.

    Returns:
        List[torch.nn.ParameterDict]: A list of ParameterDict objects representing splats.
    """
    serialized_splats = torch.load(path)
    return [
        torch.nn.ParameterDict(
            {
                key: torch.nn.Parameter(torch.tensor(value, device=device))
                for key, value in splat.items()
            }
        )
        for splat in serialized_splats
    ]


def load_splats_hdf5(path: Path, device: torch.device) -> List[torch.nn.ParameterDict]:
    """
    Load a list of splats (ParameterDict) from an HDF5 file.

    Args:
        path (Path): The file path to load the splats.
        device (torch.device): The device to load the splats onto.

    Returns:
        List[torch.nn.ParameterDict]: A list of ParameterDict objects representing splats.
    """
    splat_list = []
    with h5py.File(path, "r") as f:
        for group_name in f.keys():
            group = f[group_name]
            splat_list.append(
                torch.nn.ParameterDict(
                    {
                        key: torch.nn.Parameter(
                            torch.tensor(group[key][:], device=device)
                        )
                        for key in group.keys()
                    }
                )
            )
    return splat_list


def save_splat(splat: torch.nn.ParameterDict, path: Path) -> None:
    """
    Save a single splat (ParameterDict) to a Torch file.

    Args:
        splat (torch.nn.ParameterDict): A ParameterDict object representing the splat.
        path (Path): The file path to save the splat.
    """
    splat_dict = {key: value.detach().cpu() for key, value in splat.items()}
    torch.save(splat_dict, path)


def save_splat_hdf5(splat: torch.nn.ParameterDict, path: Path) -> None:
    """
    Save a single splat (ParameterDict) to an HDF5 file.

    Args:
        splat (torch.nn.ParameterDict): A ParameterDict object representing the splat.
        path (Path): The file path to save the splat.
    """
    # @Rok added swmr to ensure sync + properly closed f
    with h5py.File(path, "w", swmr=True) as f:
        for key, value in splat.items():
            f.create_dataset(key, data=value.detach().cpu().numpy())
            f.close()


def save_splats(splat_list: List[torch.nn.ParameterDict], path: Path) -> None:
    """
    Save a list of splats (ParameterDict) to a Torch file.

    Args:
        splat_list (List[torch.nn.ParameterDict]): A list of ParameterDict objects representing splats.
        path (Path): The file path to save the splats.
    """
    serialized_splats = [
        {key: value.detach().cpu().numpy() for key, value in splat.items()}
        for splat in splat_list
    ]
    torch.save(serialized_splats, path)


def save_splats_hdf5(splat_list: List[torch.nn.ParameterDict], path: Path) -> None:
    """
    Save a list of splats (ParameterDict) to an HDF5 file.

    Args:
        splat_list (List[torch.nn.ParameterDict]): A list of ParameterDict objects representing splats.
        path (Path): The file path to save the splats.
    """
    with h5py.File(path, "w", swmr=True) as f:
        for i, splat in enumerate(splat_list):
            group = f.create_group(f"splat_{i}")
            for key, value in splat.items():
                group.create_dataset(key, data=value.detach().cpu().numpy())
