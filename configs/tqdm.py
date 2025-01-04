from tqdm import tqdm


def get_progress_bar(iterable, max_steps=None, description="Progress", color="cyan"):
    """
    Returns a configured tqdm progress bar.

    Args:
        iterable (iterable): The iterable to wrap.
        max_steps (int, optional): Maximum number of steps.
        description (str, optional): Description for the progress bar.
        color (str, optional): Color of the progress bar.

    Returns:
        tqdm: Configured tqdm progress bar.
    """
    return tqdm(
        iterable,
        desc=description,
        total=max_steps,
        unit="step",
        colour=color,
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )
