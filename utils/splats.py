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


def merge_spherical_harmonics(splat: torch.nn.ParameterDict) -> torch.nn.ParameterDict:
    """
    Merges the spherical harmonics coefficients (sh0 and shN) into a single 'color' tensor
    and removes the original 'sh0' and 'shN' keys from the ParameterDict.

    Args:
        splat (torch.nn.ParameterDict): A dictionary of parameters containing optional 'sh0' and 'shN' keys.

    Returns:
        torch.nn.ParameterDict: Updated ParameterDict with a 'color' key if 'sh0' and 'shN' are present,
                                otherwise raises an error or ensures consistency.
    """
    if "sh0" in splat and "shN" in splat:
        # Combine sh0 and shN into a single tensor 'color'
        splat["color"] = torch.cat(
            [splat["sh0"], splat["shN"]],
            dim=1,
        ).float()
        del splat["sh0"]
        del splat["shN"]
    else:
        # Clean up any partial keys and ensure consistency
        keys_removed = []
        for key in ["sh0", "shN"]:
            if key in splat:
                del splat[key]
                keys_removed.append(key)

        if "color" not in splat:
            raise ValueError(
                f"Cannot create 'color' tensor because required keys are missing: {', '.join(keys_removed)}"
                if keys_removed
                else "No valid spherical harmonics coefficients found in the splat."
            )

    return splat


def generate_random_splat(num_points):
    """
    Generate random splat with specified shapes.

    Args:
        num_points (int): Number of points to generate.

    Returns:
        tuple: Means, quats, scales, opacities, colors, viewmats, ks, sh0, shN.
    """
    means = torch.rand(num_points, 3)
    quats = torch.rand(num_points, 4)
    quats = quats / torch.norm(quats, dim=1, keepdim=True)
    scales = torch.rand(num_points, 3)
    opacities = torch.rand(num_points, 1)
    colors = torch.rand(num_points, 3)
    viewmats = torch.rand(4, 4)
    ks = torch.rand(3, 3)
    sh0 = torch.rand(num_points, 1)
    shN = torch.rand(num_points, 2)

    return means, quats, scales, opacities, colors, viewmats, ks, sh0, shN
