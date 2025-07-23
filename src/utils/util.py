import numpy as np
import torch

from src.utils.constants import A_MIN, A_MAX


def rand_int(lower, upper_exclusive):
    """
    Generate a random number from a uniform distribution constrained to [lower, upper_exclusive).

    Args:
        lower: lower bound.
        upper_exclusive: upper exclusive bound.

    Returns:
        random integer from [lower, upper_exclusive).
    """
    return np.random.randint(lower, upper_exclusive)


def get_input(batch, **keys):
    """
    Get the inputs from a batch dictionary. If one of the keys is not in batch, then None is returned for
    that specific key.

    Args:
        batch: dictionary containing at least an image_key.
        keys: keys with which to extract from the batch.

    Returns:
        Either a single image batch of shape BCHW(D) or a tuple of image and other conditioning batches.
    """
    items = [batch.get(k, None) for k in keys.values()]
    return tuple(items) if len(items) > 1 else items[0]


def rgb_colorlist(num_colors: int):
    """
    Picks the specified amount of colors from a linear RGB colorspace.
    Based on https://stackoverflow.com/a/50980960.

    Args:
        num_colors: The amount of colors.

    Returns:
        np array of RGB colors.
    """
    t = np.linspace(-510, 510, num_colors - 1)
    colors = np.round(np.clip(np.stack([-t, 510 - np.abs(t), t], axis=1), 0, 255)).astype(np.uint8).tolist()
    # set first background color to dark gray
    return [[50, 50, 50]] + colors


def normalize_grayscale(x: np.ndarray, vmin=A_MIN, vmax=A_MAX):
    """
    Normalizes and clips an output image from the model to grayscale [0,1].

    Args:
        x: input image of shape 1HW(D).
        vmin: Lower range of processed image bound.
        vmax: Upper range of processed image bound.

    Returns:
        np.array of shape 1HW(D)
    """
    return np.clip((x - vmin) / (vmax - vmin), a_min=0.0, a_max=1.0)


def grayscale_to_rgb(img: np.ndarray):
    """
    Converts a grayscale image to an RGB colorspace image.

    Args:
        img: grayscale image of shape 1HW(D).

    Returns:
        np.array of shape 3HW(D)
    """
    img = 255 * img
    img = np.array([3 * (g,) for g in img.flatten()],
                   dtype=np.uint8).reshape(*img.shape[1:], 3)
    return np.moveaxis(img, -1, 0)


def one_hot_to_rgb(x: np.ndarray, axis: int = 0, is_one_hot: bool = True):
    """
    Creates an image of a one-hot encoded semantic map. Input is LHW(D) where L is number of semantic classes.

    Args:
        x: input one-hot encoded semantic map.
        axis: one-hot encoded dimension.
        is_one_hot: whether semantic map is one_hot.

    Returns:
        numpy array of shape 3HW(D)
    """
    colors = rgb_colorlist(x.shape[0] if is_one_hot else int(x.max()) + 1)

    if is_one_hot:
        x = np.array([colors[i] for i in np.argmax(x, axis=axis).flatten()],
                    dtype=np.uint8).reshape(*x.shape[1:], 3)
    else:
        x = np.array([colors[int(i)] for i in x.flatten()],
                    dtype=np.uint8).reshape(*x.shape[1:], 3)

    return np.moveaxis(x, -1, 0)


def pad_image(x: np.ndarray, value: float = 0.0):
    """
    Pads an image of shape 1WH[D] in symmetric fashion to a square/cube with constant values.

    Args:
        x: input image.
        value: constant value with which to fill.

    Returns:
        numpy array of shape 1SS[S].
    """
    spatial_size = max(x.shape[1:])
    # do not pad channel dim
    pad_widths = [(0, 0)] + [((spatial_size - s) // 2, (spatial_size - s) // 2 + (s & 1)) for s in x.shape[1:]]
    return np.pad(x, pad_widths, constant_values=value)


def cosine_warmup_lr_scheduler(optimizer, max_steps: int, warmup_steps: int, lr: float, min_lr: float):
    """
    Returns a cosine learning scheduler with linear warmup, and constant minimum learning rate once
    maximum number of steps is reached.

    Args:
        optimizer: torch optimizer with model parameters.
        max_steps: maximum number of steps.
        warmup_steps: the number of lr warmup steps.
        lr: maximum learning rate value, reached after warmup steps.
        min_lr: minimum learning rate, reached after specified maximum number of steps.
    """

    # simulate cosine annealing with linear warmup and constant learning rate after reaching learning_rate_min
    T_max = max_steps - warmup_steps
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer=optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.0001, end_factor=1.0, total_iters=warmup_steps),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=min_lr),
            torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _ : min_lr / lr),
        ],
        milestones=[warmup_steps, max_steps])

    return lr_scheduler


def pos_enc_3d(coords: torch.Tensor, embedding_size: int):
    """
    Compute 3D positional encoding. Code inspired by
    https://github.com/tatp22/multidim-positional-encoding/blob/master/positional_encodings/torch_encodings.py.

    Args:
        coords: coordinates of shape Nx3.

    Returns:
        tensor of shape Nx[pos_enc_emb]
    """
    dtype = coords.dtype
    coords = coords.to(dtype=torch.float64)
    # single coord dim
    xyz_dim = embedding_size // 3
    inv_freqs = 1.0 / (10000 ** (torch.arange(0, xyz_dim, 2, dtype=torch.float64) / xyz_dim))
    xyz_embs = []

    for xyz in [0, 1, 2]:
        sinusoid_input = torch.einsum("i,j->ij", coords[:, xyz], inv_freqs)
        xyz_emb = torch.cat((sinusoid_input.sin(), sinusoid_input.cos()), dim=-1)
        xyz_embs.append(xyz_emb)

    out = torch.cat(xyz_embs, dim=-1)
    return out.to(dtype=dtype)
