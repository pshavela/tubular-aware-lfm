from __future__ import annotations

import torch

from src.utils.util import pad_image, pos_enc_3d
from src.processing.autoencode import VAEProcessor
from typing import Tuple, Dict
from skimage.morphology import skeletonize
from scipy.ndimage import binary_dilation, distance_transform_edt
from monai import transforms
from monai.data.meta_tensor import MetaTensor
from monai.utils.misc import ensure_tuple_rep
from monai.config.type_definitions import KeysCollection
from monai.transforms import RandomizableTransform, MapTransform, Orientationd


class DimensionAwareOrientationd(Orientationd):
    """
    Performs axcode-specified orientation transform for 3D images, 2D images left as is.
    """
    def __init__(self, keys: KeysCollection, dim: int, **kwargs):
        super().__init__(keys, **kwargs)
        self.dim = dim

    def __call__(self, data, **kwargs):
        if self.dim == 2:
            return dict(data)
        return super().__call__(data, **kwargs)


class EncodeToLatentsTransformd(MapTransform):
    """
    Encode images/labels to the latent representations with a pretrained AE.
    Latents are moved back on CPU device.
    """
    def __init__(self, keys: KeysCollection, type: str = 'image', allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        assert type in ['image']
        self.type = type

    def __call__(self, data):
        d = dict(data)

        for key in self.key_iterator(d):
            if self.type == 'image':
                z = VAEProcessor.encode(d[key].unsqueeze(0))
                d[key] = z.squeeze(0).to('cpu')

        return d


class CenterSliceTransform(MapTransform):
    """
    Pick the middle slice in each spatial direction of a cube grayscale image of shape 1RR[R]. Returns shape 3RR
    for 3D images, and 1RR for 2D.
    """
    def __init__(
        self,
        mapping: Dict[str, str]
    ):
        super().__init__(list(mapping.keys()), allow_missing_keys=True)
        self.mapping = mapping

    def __call__(self, data):
        d = dict(data)

        for key, output_key in self.mapping.items():
            if key in d:
                image = d[key]
                # pad in case image is not square/cube
                image = torch.from_numpy(pad_image(image))

                if len(image.shape) == 4:
                    # 3D
                    _, r, *_ = image.shape
                    image = torch.cat([image[:, r // 2, :, :],
                                        image[:, :, r // 2, :],
                                        image[:, :, :, r // 2]])
                d[output_key] = image

        return d


class DilateBinaryLabelMapTransformd(MapTransform):
    """
    Dilate a binary label map, eg. for vessels.
    """
    def __init__(
            self,
            keys: KeysCollection,
            iterations: int = 0,
            allow_missing_keys: bool = False
        ):
        super().__init__(keys, allow_missing_keys)
        self.iterations = iterations

    def __call__(self, data):
        d = dict(data)


        for key in self.key_iterator(d):
            binary_map = d[key]
            if self.iterations > 0:
                out = binary_dilation(binary_map, iterations=self.iterations)
                out = torch.from_numpy(out).to(dtype=torch.float32)
            else:
                out = binary_map.to(dtype=torch.float32)

            d[key] = out

        return d


class SplitVesselLabelTransform(MapTransform):
    def __init__(
        self,
        label_key: str = 'label',
        output_key: str = 'detail',
        do_transform: bool = True,
        allow_missing_keys: bool = False
    ):
        super().__init__(
            keys=[label_key],
            allow_missing_keys=allow_missing_keys or not do_transform
        )
        self.label_key = label_key
        self.output_key = output_key
        self.do_transform = do_transform

    def __call__(self, data):
        data = dict(data)

        if self.do_transform:
            label: MetaTensor = data[self.label_key]
            detail = (label == label.max()).to(torch.uint8)
            data[self.output_key]  = detail

        return data


class EncodeVesselsTransformd(MapTransform):
    """
    extract centerline from a binary vessel label map, collect all coordinates c=(x,y,z) along centerline
    and compute radius for each c. If desired, normalize coordinates to [0, 1]. Standardize radius to
    voxels. If desired, create positional encoding embedding matrix where each row is positional embedding
    vector with radius concatenation. 3D Positional encoding extended to 3D from original
    transformer paper "Attention is all you need" (https://arxiv.org/pdf/1706.03762) by concatenating each
    sinusoidal embedding of each coordinate point x, y, z.

    NOTE: Only for 3D images.

    Args:
        pad_size: size of final tensor of shape Px[...], can be None if variable sequence length is desired.
        pos_enc_emb: if not None, the dimension of the sinusoidal positional encoding vectors.
        normalize_unit: whether to normalize coordinates to [0, 1].
        store_coords: whether to store the coordinates as a separate key entry before positional encoding.

    Returns:
        tensor of shape Px[pos_enc_emb] where radius is concated at the last dimension
            if pos_enc_emb is not none, else Px4 containing [x, y, z, radius] rows.
    """
    def __init__(
            self,
            keys,
            pad_size: int | None = None,
            pos_enc_emb: int | None = None,
            normalize_unit: bool = False,
            store_coords: bool = False,
            allow_missing_keys = False
    ):
        super().__init__(keys, allow_missing_keys)
        self.pad_size = pad_size
        self.normalize_unit = normalize_unit
        self.store_coords = store_coords

        if pos_enc_emb is not None:
            assert pos_enc_emb > 0 and pos_enc_emb % 6 == 0, '3D sinusoidal embedding dim must be divisible by 6.'
            self.pos_enc_emb = pos_enc_emb

    def __call__(self, data):
        data = dict(data)

        for key in self.key_iterator(data):
            vessel_map: MetaTensor = data[key]
            vessel_map = vessel_map.squeeze()

            centerline = torch.from_numpy(skeletonize(vessel_map))
            # Nx3
            coords = torch.argwhere(centerline)

            distances = distance_transform_edt(vessel_map)
            # N
            radius = distances[coords[:, 0], coords[:, 1], coords[:, 2]]
            radius = torch.from_numpy(radius)

            if self.normalize_unit:
                # normalize coordinates
                coords = coords / torch.tensor(vessel_map.shape).to(coords)
                # normalize radius
                radius /= torch.tensor(vessel_map.shape, dtype=torch.float32).mean()

            tokens = coords
            if self.pos_enc_emb > 0:
                tokens = pos_enc_3d(tokens, self.pos_enc_emb)

            # Nx4
            tokens = torch.cat((tokens, radius.unsqueeze(-1)), dim=1)

            if self.pad_size is not None:
                # Px[..], pad rows bottom
                tokens = torch.nn.functional.pad(tokens, (0, 0, 0, self.pad_size - tokens.shape[0]))

            # to float32
            tokens = tokens.to(dtype=torch.float32)
            data[key] = tokens

            if self.store_coords:
                coords = torch.cat((coords, radius.unsqueeze(-1)), dim=1)
                data[f'{key}_coords'] = coords.to(dtype=torch.float32)

        return data


class OneHotTransformd(RandomizableTransform, MapTransform):
    """
    Transform semantic label map from scalar to one-hot encoding.
    RandomizableTransform is inherited in order to trick the MONAI dataset pipeline into not preprocessing this transform
    and execute it on the fly - the idea here is to save on storage when using eg. MONAI's PersistentDataset or
    CacheDataset, whereby one-hot encoded tensors take up much more space.
    """
    def __init__(self, keys: KeysCollection, number_classes: int, allow_missing_keys: bool = False):
        RandomizableTransform.__init__(self)
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.number_classes = number_classes

    def __call__(self, data):
        d = dict(data)

        for key in self.key_iterator(d):
            semantic_map = d[key]
            d[key] = transforms.AsDiscrete(to_onehot=self.number_classes)(semantic_map)

        return d


class RandRotate90AllAxisd(MapTransform, RandomizableTransform):
    """
    Randomly rotate by 90 degrees along any of the available spatial axis.
    """
    def __init__(self, keys: KeysCollection, dim: int, prob: float = 0.1, allow_missing_keys: bool = False,
                 do_transform: bool = True):
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob, do_transform)
        self.transforms = [transforms.Rotate90(spatial_axes=(0, 1))]
        if dim == 3:
            self.transforms += [
                transforms.Rotate90(spatial_axes=(0, 2)),
                transforms.Rotate90(spatial_axes=(1, 2)),
            ]

    def __call__(self, data):
        d = dict(data)

        do_transforms = []
        for _ in self.transforms:
            self.randomize(None)
            do_transforms.append(self._do_transform)

        for key in self.key_iterator(d):
            for do_transform, transform in zip(do_transforms, self.transforms):
                if do_transform:
                    d[key] = transform(d[key])

        return d


class RandLabelDropd(RandomizableTransform, MapTransform):
    """
    Randomly fill out the whole semantic label map with 0s.
    """
    def __init__(self, keys: KeysCollection, prob: float = 0.2, allow_missing_keys: bool = False,
                 do_transform: bool = True):
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob, do_transform)

    def randomize(self):
        super().randomize(None)

    def __call__(self, data):
        d = dict(data)
        self.randomize()

        for key in self.key_iterator(d):
            if self._do_transform:
                img = d[key]
                d[key] = torch.zeros_like(img)

        return d


class SpacingResized(MapTransform):
    """
    Transforms the image by first resizing with the specified spacing in all spatial dimensions and then
    pads/crops to the specified resolution in all spatial dimensions. If spacing is None, then the image
    is resized to the specified resolution with an Affine transformation.

    Args:
        resolution: specified resolution of all spatial dimensions.
        spacing: specified spacing of all spatial dimensions.
        mode: one of ['image', 'label']. determines the spline interpolation order
            during resizing, which is 3 for images and 0 for labels.
        pad_constant: padding constant.
    """
    def __init__(
            self,
            keys: KeysCollection,
            resolution: int | Tuple[int, int] | Tuple[int, int, int],
            spacing: None | float | Tuple[float, float] | Tuple[float, float, float],
            mode: str,
            pad_constant: int,
            allow_missing_keys: bool = False
        ):
        super().__init__(keys, allow_missing_keys)
        self.resolution = resolution
        self.spacing = spacing
        assert mode in ['image', 'label']
        self.mode = mode
        self.pad_constant = pad_constant

    def __call__(self, data):
        d = dict(data)

        for key in self.key_iterator(d):
            img = d[key]
            d[key] = self._transform(img)

        return d

    def _transform(self, image):
        """
        Transforms the image with the specified spacing and pads to resolution in all spatial dimensions.

        Args:
            image: The 2/3D image as input.

        Returns:
            Transformed image.
        """
        dim = len(image.shape) - 1
        spatial_size = ensure_tuple_rep(self.resolution, dim)

        if dim == 3:
            if not self.spacing:
                transformed = resize_scale(image, spatial_size, self.mode)
            else:
                spacing = ensure_tuple_rep(self.spacing, dim)
                transformed = transforms.Compose([
                    transforms.Spacing(pixdim=spacing, mode=interpolation_order(self.mode)),
                    transforms.ResizeWithPadOrCrop(spatial_size=spatial_size, value=self.pad_constant)
                ])(image)
        else:
            transformed = resize_scale(image, spatial_size, self.mode)

        return transformed


class PopulateSpacingTransform(MapTransform):
    """
    Updates spacing information of an image. Requires keys 'image' and 'spacing' in the dictionary of each
    data element.

    Args:
        resolution: new resolution of all spatial dimensions.
        spacing: specified spacing of all spatial dimensions.
        image_key: key with which to retrieve the image from which the spacing can be extracted in case spacing
            is not specified.
        spacing_key: key whose field will be populated with the spacing value.
    """
    def __init__(self,
                 resolution: int | Tuple[int, int] | Tuple[int, int, int],
                 spacing: float | Tuple[float, float] | Tuple[float, float, float] = None,
                 image_key: str = 'image',
                 spacing_key: str = 'spacing'
        ):
        super().__init__(keys=[image_key, spacing_key], allow_missing_keys=False)
        self.resolution = resolution
        self.spacing = spacing
        self.image_key = image_key
        self.spacing_key = spacing_key

    def __call__(self, data):
        """
        Updates the spacing of the MONAI metatensor of a 2/3D image. For 2D images a constant line spacing of 1.0 is
        returned in case spacing is not specified.
        """
        d = dict(data)
        image = d[self.image_key]
        dim = len(image.shape) - 1

        if self.spacing:
            spacing = ensure_tuple_rep(self.spacing, dim)
            spacing = torch.FloatTensor(spacing)
        else:
            if dim == 3:
                old_spacings = torch.FloatTensor(image.meta['pixdim'][1:4])
                old_size = torch.tensor(image.shape[1:])
                new_size = ensure_tuple_rep(self.resolution, dim)
                new_size = torch.FloatTensor(new_size)
                spacing = (old_size / new_size) * old_spacings
            else:
                spacing = torch.FloatTensor([1.0, 1.0])

        d[self.spacing_key] = spacing
        return d


def resize_scale(img, resolution: Tuple[int, int] | Tuple[int, int, int], mode: str = 'image'):
    """
    Resizes the input image to the defined resolution on all spatial dimension whilst scaling appropriately.

    Args:
        img: The input image of shape CxHxWxD (or CxHxW for 2D).
        resolution: the defined output spatial resolution.
        mode: One of ['image', 'label']. Specifies the internal spline interpolation order for the interpolation.

    Returns:
        The transformed image.
    """
    if all(s == r for s, r in zip(img.shape[1:], resolution)):
        return img

    dim = len(img.shape) - 1
    if dim == 3:
        _, h, w, d = img.shape
        return transforms.Affine(scale_params=(h / resolution[0], w / resolution[1], d / resolution[2]),
                                 spatial_size=resolution,
                                 padding_mode='border',
                                 mode=interpolation_order(mode),
                                 image_only=True)(img)
    else:
        _, h, w = img.shape
        return transforms.Affine(scale_params=(h / resolution[0], w / resolution[1]),
                                 spatial_size=resolution,
                                 padding_mode='border',
                                 mode=interpolation_order(mode),
                                 image_only=True)(img)


def interpolation_order(mode: str = 'image'):
    return {
        'image' : 3,
        'label': 0
    }[mode]
