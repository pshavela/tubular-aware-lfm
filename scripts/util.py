from __future__ import annotations

import numpy as np
import torch

from src.utils.util import normalize_grayscale
from src.utils.constants import VOLUME
from monai import transforms
from monai.data.image_writer import NibabelWriter, PILWriter


def write_image(
        img: torch.Tensor,
        path: str,
        affine: torch.Tensor = None,
        is_onehot: bool = False,
        denormalize_intensity: bool = True,
        verbose: bool = True
    ):
    """
    """
    img = img.detach().cpu().numpy()

    if is_onehot:
        denormalize_intensity = False
        img = np.argmax(img, axis=0)
        img = np.expand_dims(img, 0)

    if len(img.shape) == 3:
        # 2D
        if denormalize_intensity:
            img = normalize_grayscale(img)

        writer = PILWriter(output_dtype=np.uint8, scale=255 if denormalize_intensity else None)
        writer.set_data_array(img)
    else:
        # 3D
        if denormalize_intensity:
            img = transforms.ScaleIntensityRange(a_min=VOLUME.a_min, a_max=VOLUME.a_max,
                                                 b_min=VOLUME.clip_min, b_max=VOLUME.clip_max,
                                                 clip=True, dtype=np.int16)(img)
        writer = NibabelWriter(output_dtype=np.int16)
        writer.set_data_array(img)
        if affine is not None:
            writer.set_metadata(meta_dict={'affine': affine})

    writer.write(path, verbose=verbose)


def gpu_vram_usage(device) -> float:
    if device == torch.device('cpu'):
        return 0

    free, total = torch.cuda.mem_get_info(device)
    vram_used_gb = (total - free) / (1024 ** 3)
    return vram_used_gb
