import numpy as np

from typing import List
from src.utils.util import normalize_grayscale, grayscale_to_rgb, one_hot_to_rgb, pad_image


class LoggerWrapper:
    """
    Wrapper for WandBLogger/TensorBoardLogger.

    Args:
        logger: wandb/tensorboard logger instance.
        logger_type: logger type
    """
    def __init__(self, logger, logger_type: str):
        assert logger_type in ['wandb', 'tensorboard']
        self.logger_type = logger_type
        self.logger = logger.experiment if logger_type == 'tensorboard' else logger

    def log_image_pair(
            self,
            real: np.ndarray,
            fake: np.ndarray,
            prefix: str = '',
            caption: str = '',
            step: int = None,
            to_one_hot: bool = False,
            full_real: bool = True,
    ):
        """
        Log the middle slices along each spatial dimension of a single image pair.

        Args:
            real: real image with shape 1HW(D).
            fake: fake image with shape 1HW(D).
            prefix: prefix key.
            caption: caption for WandB.
            step: current global step.
            to_one_hot: whether images are one-hot, eg. for labels.
            full_real: whether the real image has full shape, or just the center slices.
        """
        if not full_real:
            real = np.expand_dims(real, 0)
        else:
            # pad in case images are not square/cube
            real = pad_image(real)

        fake = pad_image(fake)

        # normalize images to grayscale
        if not to_one_hot:
            real = normalize_grayscale(real)
            fake = normalize_grayscale(fake)
        else:
            real = one_hot_to_rgb(real)
            fake = one_hot_to_rgb(fake)

        if len(real.shape) == 4:
            # 3D, create a middle slice along each spatial dimension and join each image-pair vertically
            _, h, w, d = real.shape
            if full_real:
                grid = [
                    np.concatenate((real[:, h // 2, :, :], fake[:, h // 2, :, :]), axis=1),
                    np.concatenate((real[:, :, w // 2, :], fake[:, :, w // 2, :]), axis=1),
                    np.concatenate((real[:, :, :, d // 2], fake[:, :, :, d // 2]), axis=1),
                ]
            else:
                grid = [
                    np.concatenate((real[:, 0, :, :], fake[:, h // 2, :, :]), axis=1),
                    np.concatenate((real[:, 1, :, :], fake[:, :, w // 2, :]), axis=1),
                    np.concatenate((real[:, 2, :, :], fake[:, :, :, d // 2]), axis=1),
                ]
        else:
            # 2D, join single image-pair horizontally along width dimension
            grid = [real, fake]

        if self.logger_type == 'wandb':
            if to_one_hot:
                # wandb expects last dim as channel
                grid = [np.moveaxis(img, 0, -1) for img in grid]
            self.logger.log_image(key=prefix, images=grid, caption=len(grid) * [caption])
        else:
            self.logger.add_images(f'{prefix}', np.array(grid), global_step=step)

    def log_conditional_image(
            self,
            real: np.ndarray,
            synthetic: np.ndarray,
            label: np.ndarray,
            prefix: str = 'conditional',
            caption: str = '',
            is_slices: bool = True,
            step: int = None,
        ):
        """
        Log the middle slices along each spatial dimension of a single image, and the semantic label map.

        Args:
            real: real image based on semantic label map with shape 1HW(D) if full_real, else 3HW.
            synthetic: sampled image with shape 1HW(D).
            label: semantic label map with shape LHW(D), where L is number of label classes.
            prefix: prefix for tensorboard/wandb.
            caption: caption to WandB.
            is_slices: whether the real image and label have full shape, or just the center slices.
            step: current global step.
        """

        if is_slices:
            real = np.expand_dims(real, 0)
            label = np.expand_dims(label, 0)
        else:
            # pad in case images are not square/cube
            real = pad_image(real)
            label = pad_image(label)

        synthetic = pad_image(synthetic)

        # convert one-hot encoding to image of shape 3HW(D)
        label = one_hot_to_rgb(label, is_one_hot=label.shape[0] > 1)
        # convert images to grayscale and of shape 1HW(D) to 3HW(D)
        real = grayscale_to_rgb(normalize_grayscale(real))
        synthetic = grayscale_to_rgb(normalize_grayscale(synthetic))

        if synthetic.shape[-1] != real.shape[-1]:
            # if any spatial dimension of the real and synthetic does not match then the latent resolution is probably too high/low.
            raise Exception(f'Spatial dimension of synthetic ({synthetic.shape}) and real ({real.shape}) images do not match!'
                             + ' Most likely the `latent_resolution` in the model config is ill defined.')

        if len(synthetic.shape) == 4:
            # 3D, create a middle slice along each spatial dimension and join each image-label triple horizontally
            _, h, w, d = synthetic.shape
            if not is_slices:
                grid = [
                    np.concatenate((real[:, h // 2, :, :], label[:, h // 2, :, :], synthetic[:, h // 2, :, :]), axis=1),
                    np.concatenate((real[:, :, w // 2, :], label[:, :, w // 2, :], synthetic[:, :, w // 2, :]), axis=1),
                    np.concatenate((real[:, :, :, d // 2], label[:, :, :, d // 2], synthetic[:, :, :, d // 2]), axis=1),
                ]
            else:
                grid = [
                    np.concatenate((real[:, 0, :, :], label[:, 0, :, :], synthetic[:, h // 2, :, :]), axis=1),
                    np.concatenate((real[:, 1, :, :], label[:, 1, :, :], synthetic[:, :, w // 2, :]), axis=1),
                    np.concatenate((real[:, 2, :, :], label[:, 2, :, :], synthetic[:, :, :, d // 2]), axis=1),
                ]
        else:
            # 2D, join single image-label triple horizontally along height dimension
            grid = [real, label, synthetic]

        if self.logger_type == 'wandb':
            # wandb expects last dim as channel
            grid = [np.moveaxis(img, 0, -1) for img in grid]
            self.wnb.log_image(key=prefix, images=grid, caption=len(grid) * [caption])
        else:
            self.logger.add_images(prefix, np.array(grid), global_step=step)

    def log_sampling_process(
            self,
            synthetics: List[np.ndarray],
            caption: str = '',
            step: int = None,
        ):
        """
        Log the middle slices along each spatial dimension of a single image throughout the diffusion/flow process.

        Args:
            synthetics: The list of intermediate images and the final image of a sampling process.
            caption: caption to WandB.
        """

        synthetics = [normalize_grayscale(s) for s in synthetics]
        # pad
        synthetics = [pad_image(s) for s in synthetics]

        if len(synthetics[0].shape) == 4:
            # 3D, create a middle slice along each spatial dimension and join each image-label triple horizontally
            _, h, w, d = synthetics[0].shape
            grid = [np.concatenate((s[:, h // 2, :, :], s[:, :, w // 2, :], s[:, :, :, d // 2]), axis=1)
                    for s in synthetics]
        else:
            # 2D, join single image-label triple horizontally along height dimension
            grid = synthetics

        grid = [np.concatenate(grid, axis=2)]

        if self.logger_type == 'wandb':
            self.wnb.log_image(key='process', images=grid, caption=len(grid) * [caption])
        else:
            self.logger.add_images(f'process', np.array(grid), global_step=step)
