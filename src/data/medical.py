from __future__ import annotations

import os
import glob
import torch
import lightning as L

from src.utils.constants import IMAGE, VOLUME
from src.data.utils import (
    RandRotate90AllAxisd,
    RandLabelDropd,
    SpacingResized,
    PopulateSpacingTransform,
    EncodeToLatentsTransformd,
    OneHotTransformd,
    CenterSliceTransform,
    DimensionAwareOrientationd,
    DilateBinaryLabelMapTransformd,
    SplitVesselLabelTransform,
    EncodeVesselsTransformd
)
from typing import Tuple
from tqdm import tqdm
from monai import transforms
from monai.utils.misc import ensure_tuple_rep
from monai.data import Dataset, PersistentDataset, CacheDataset, DataLoader


class MedicalDataModule(L.LightningDataModule):
    """
    DataModule for loading 2D/3D grayscale medical images. Images in the respective directories must have a numbered
    scheme, ie. images must start with a number, eg.
        '1.img.nii.gz' or '1.img.png'
        and
        '1.label.nii.gz' or '1.label.png'
    so that a proper alignment of the data can be guaranteed.
    Training and Validation images must locate in separate directories, same with input images and labels.

    Args:
        dim: dimension of images, one of [2, 3].
        resolution: spatial size in all 3 dimensions after spacing transformation.
        image_train_dir: directory path containing training images.
        image_val_dir: directory path containing validation images.
        label_train_dir: directory path containing labels for training images.
        label_val_dir: directory path containing labels for validation images.
        detail_train_dir: directory path containing binary labels such as vessels for training.
        detail_val_dir: directory path containing binary labels for validation.
        split_label_detail: Split key `label` into `label` and `detail`, whereby detail will become a binary label map
            of the maximum class value in label. Can be used to avoid loading detail_[train/val]_dir, in case detail
            class is also present in the label maps.
        split_label_vessel: identical to split_label_detail, will be stored in `vessel`.
        dilate_detail_iterations: number of times to dilate the detail binary maps.
        dilate_vessel_iterations: identical to `dilate_detail_iterations`, but for `vessel` key.
        number_classes: The possible number of classes in the semantic label map.
        spacing: The amount of spacing on each spatial plane, must be a single floating number or None.
        encode_latents: When to encode the images with a pretrained VQVAE. Can be one of three modes:
            ['pre', 'post', none].
            - 'pre': will pre-encode for faster training but wont allow for augmentations
            - 'post': will post-encode after all random transformations are done, but will be slower
            - none: no encoding will be done
        encode_detail_pos: whether to encode the centerline coordinate of the vessels.
        encode_detail_pos_coords: whether to store the coordinates and radius separately to the encoding.
        encode_detail_pos_size: padding size for vessel centerline coordinate/diameter pairs.
        encode_detail_pos_emb_size: whether to also apply 3D sinusoidal positional encoding for vessel coordinates,
            determines the embedding size
        keep_full_original: In combination with encode_latents, determines whether to supply the original image full size
            for stage 2 training. Should be False in order to minimize resource usage, the original image is only used
            during image logging as a comparison, only 2/3 slices necessary for each spatial dimension.
        augment: Whether to augment the data with random rotations, intensity changes and flips.
        random_crop: Whether to apply random crop during training. Specifies the region of interest size.
        label_one_hot: Whether to convert the label map to a one-hot encoding.
        label_drop_prob: When augment is True, randomly zero out the labels with a predefined probability.
        label_downsample: Determines the resolution of further downsampling operations, eg. to latent resolution.
        vessel_downsample: Same as label_downsample, only for `vessel` key.
        dtype: output dtypes for all dataset tensors.
        dataset_type: One of [default, cache, persistent]. Determines whether to use MONAI's default Dataset (ran on
            the fly), CacheDataset (speeds up training by preprocessing), or PersistentDataset (preprocessed storing).
        persistent_dir: where to store preprocessed files, for PersistentDataset.
        batch_size: Batch size.
        num_workers: Number of CPU workers for CacheDataset.
        device: device on which to transfer the process. Some operations might require GPU access.
    """
    def __init__(self,
                 *,
                 dim: int,
                 resolution: int | Tuple[int, int] | Tuple[int, int, int],
                 spacing: None | float | Tuple[float, float] | Tuple[float, float, float] = None,
                 image_train_dir: str = '',
                 image_val_dir: str = '',
                 label_train_dir: str = '',
                 label_val_dir: str = '',
                 detail_train_dir: str = '',
                 detail_val_dir: str = '',
                 split_label_detail: bool = False,
                 split_label_vessel: bool = False,
                 dilate_detail_iterations: int = 0,
                 dilate_vessel_iterations: int = 0,
                 smooth_detail: bool = False,
                 number_classes: int = None,
                 encode_latents: str = None,
                 encode_detail_pos: bool = False,
                 encode_detail_pos_coords: bool = False,
                 encode_detail_pos_normalize: bool = True,
                 encode_detail_pos_size: int | None = None,
                 encode_detail_pos_emb_size: int | None = None,
                 keep_full_original: bool = False,
                 augment: bool = True,
                 random_crop: None | int | Tuple[int, int] | Tuple[int, int, int] = None,
                 label_one_hot: bool = True,
                 label_drop_prob: float = 0,
                 label_downsample: None | int | Tuple[int, int] | Tuple[int, int, int] = None,
                 vessel_downsample: None | int | Tuple[int, int] | Tuple[int, int, int] = None,
                 dtype: torch.dtype = torch.float32,
                 dataset_type: str = 'default',
                 persistent_dir: str = './persistent_cache',
                 batch_size: int = 8,
                 num_workers: int = 4,
                 device = 'cpu'):
        super().__init__()
        self.dim = dim
        self.image_train_dir = image_train_dir
        self.image_val_dir = image_val_dir
        self.label_train_dir = label_train_dir
        self.label_val_dir = label_val_dir
        self.detail_train_dir = detail_train_dir
        self.detail_val_dir = detail_val_dir
        if label_train_dir and label_one_hot:
            assert number_classes, 'Specify number of label classes when using one-hot encoding for labels.'

        self.split_label_detail = split_label_detail
        self.split_label_vessel = split_label_vessel
        self.dilate_detail_iterations = dilate_detail_iterations
        self.dilate_vessel_iterations = dilate_vessel_iterations
        self.smooth_detail = smooth_detail
        self.number_classes = number_classes
        self.spacing = spacing
        if encode_latents not in ['pre', 'post', None]:
            raise ValueError('encode_latents must be one of [pre, post, None].')

        self.encode_latents = encode_latents
        self.encode_detail_pos = encode_detail_pos
        self.encode_detail_pos_coords = encode_detail_pos_coords
        self.encode_detail_pos_normalize = encode_detail_pos_normalize
        self.encode_detail_pos_size = encode_detail_pos_size
        self.encode_detail_pos_emb_size = encode_detail_pos_emb_size
        self.keep_full_original = keep_full_original or (self.dim == 2)
        self.augment = augment
        self.random_crop = random_crop
        self.label_one_hot = label_one_hot
        self.label_downsample = label_downsample
        self.vessel_downsample = vessel_downsample
        self.label_drop_prob = label_drop_prob
        self.resolution = resolution

        if dataset_type not in ['default', 'cache', 'persistent']:
            raise ValueError('Dataset type invalid! Must be one of [default, cache, persistent].')

        self.dtype = dtype
        self.dataset_type = dataset_type
        self.persistent_dir = persistent_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device

    def prepare_data(self, preprocess: bool = False):
        self._setup()

        if preprocess:
            for _ in tqdm(self.train_dataloader(), desc='Preprocessing training dataset...'):
                pass
            for _ in tqdm(self.val_dataloader(), desc='Preprocessing validation dataset...'):
                pass

    def _setup(self, stage = None):
        DatasetClass = Dataset

        kwargs = {}
        if self.dataset_type == 'cache':
            DatasetClass = CacheDataset
            kwargs['num_workers'] = self.num_workers
        if self.dataset_type == 'persistent':
            DatasetClass = PersistentDataset
            kwargs['cache_dir'] = self.persistent_dir

        self.train_data = self.construct_data(self.image_train_dir,
                                              self.label_train_dir,
                                              self.detail_train_dir)
        self.val_data = self.construct_data(self.image_val_dir,
                                            self.label_val_dir,
                                            self.detail_val_dir)

        keys = ['image', 'label', 'vessel', 'detail']

        # clipping ranges for normalization and padding constants
        SPECS = IMAGE if self.dim == 2 else VOLUME

        val_transforms = [
            # Load images
            transforms.LoadImaged(keys=keys, allow_missing_keys=True),
            # add channel dimension
            transforms.EnsureChannelFirstd(keys=keys, allow_missing_keys=True),
            # init orientation, only applied to 3D
            DimensionAwareOrientationd(keys=keys, dim=self.dim, axcodes='RAS', allow_missing_keys=True),
            # split label map into organs and binary vessel maps if desired
            SplitVesselLabelTransform(output_key='detail', do_transform=self.split_label_detail),
            SplitVesselLabelTransform(output_key='vessel', do_transform=self.split_label_vessel),
            # populate spacing information, either images or labels must be loaded to populate info
            PopulateSpacingTransform(self.resolution, self.spacing,
                                     image_key='image' if self.image_train_dir else ('label' if self.label_train_dir else 'detail')),
            # spatially adapt spacing and resize input image and label
            SpacingResized(keys=['image'], resolution=self.resolution, spacing=self.spacing,
                           mode='image', pad_constant=SPECS.pad_constant_image, allow_missing_keys=True),
            # labels have different spline interpolation order and padding constant
            SpacingResized(keys=['label', 'detail'], resolution=self.resolution, spacing=self.spacing,
                           mode='label', pad_constant=SPECS.pad_constant_label, allow_missing_keys=True),
            # clip image intensities and normalize
            transforms.ScaleIntensityRanged(keys=['image'],
                                            a_min=SPECS.clip_min, a_max=SPECS.clip_max,
                                            b_min=SPECS.a_min, b_max=SPECS.a_max, clip=True, allow_missing_keys=True),
            # dilate binary label map, eg. vessels
            DilateBinaryLabelMapTransformd(keys=['detail'], iterations=self.dilate_detail_iterations,
                                           allow_missing_keys=True),
            DilateBinaryLabelMapTransformd(keys=['vessel'], iterations=self.dilate_vessel_iterations,
                                           allow_missing_keys=True),
        ]

        if self.label_downsample:
            # further downsampling operation for labels
            val_transforms += [
                SpacingResized(keys=['label'], resolution=self.label_downsample, spacing=None,
                               mode='label', pad_constant=SPECS.pad_constant_label)
            ]

        if self.vessel_downsample:
            # further downsampling operation for labels
            val_transforms += [
                SpacingResized(keys=['vessel'], resolution=self.vessel_downsample, spacing=None,
                               mode='label', pad_constant=SPECS.pad_constant_label)
            ]

        if not self.random_crop and self.encode_detail_pos:
            val_transforms += [
                EncodeVesselsTransformd(keys=['detail'],
                                        pad_size=self.encode_detail_pos_size,
                                        pos_enc_emb=self.encode_detail_pos_emb_size,
                                        normalize_unit=self.encode_detail_pos_normalize,
                                        store_coords=self.encode_detail_pos_coords)
            ]

        if self.smooth_detail:
            val_transforms += [
                transforms.GaussianSmoothd(keys=['detail'], sigma=0.75),
            ]

        # we can additionally keep the center slices for image logging purposes
        val_transforms += [
            CenterSliceTransform(mapping={'image': 'original_image', 'label': 'original_label'})
        ]

        def encode_latents_transforms():
            tfs = [
                transforms.ToDeviced(keys=['image'], device=self.device),
                EncodeToLatentsTransformd(keys=['image'], type='image')
            ]

            return tfs

        if self.encode_latents == 'pre':
            val_transforms += encode_latents_transforms()

        train_transforms = val_transforms.copy()

        if self.label_one_hot:
            one_hot_last = [
                # switch to one-hot encoding for the labels
                # NOTE: this transform must be the last non-random transform. for more info read doc of OneHotTransformd
                OneHotTransformd(keys=['label'], number_classes=self.number_classes, allow_missing_keys=True),
            ]
            val_transforms += one_hot_last
            train_transforms += one_hot_last

        train_transforms += [
            # randomly drop label semantic maps
            RandLabelDropd(keys=['label', 'detail'], prob=self.label_drop_prob, allow_missing_keys=True),
        ]

        if self.random_crop is not None:
            train_transforms += [
                transforms.RandSpatialCropd(keys=['image', 'label', 'detail', 'vessel'],
                                            roi_size=ensure_tuple_rep(self.random_crop, dim=self.dim), allow_missing_keys=True),
            ]

            if self.encode_detail_pos:
                val_transforms += [
                    EncodeVesselsTransformd(keys=['detail'],
                                            pad_size=self.encode_detail_pos_size,
                                            pos_enc_emb=self.encode_detail_pos_emb_size,
                                            normalize_unit=self.encode_detail_pos_normalize,
                                            store_coords=self.encode_detail_pos_coords)
                ]
                train_transforms += [
                    EncodeVesselsTransformd(keys=['detail'],
                                            pad_size=self.encode_detail_pos_size,
                                            pos_enc_emb=self.encode_detail_pos_emb_size,
                                            normalize_unit=self.encode_detail_pos_normalize,
                                            store_coords=self.encode_detail_pos_coords)
                ]

        if self.augment:
            train_transforms += [
                # randomly flip along any spatial axis
                transforms.RandAxisFlipd(keys=['image', 'label', 'detail', 'vessel'], prob=0.5,
                                         allow_missing_keys=True),
                # randomly rotate 90 degrees up to 3 times along any spatial axis
                RandRotate90AllAxisd(keys=['image', 'label', 'detail', 'vessel'], dim=self.dim, prob=0.1,
                                     allow_missing_keys=True),
                # randomly scale intensity for image only, factor between (0.9, 1.1)
                transforms.RandScaleIntensityd(keys=['image'], factors=0.1, prob=0.1,
                                               allow_missing_keys=True),
                # randomly shift intensity for image only, offset between (-0.05, 0.05)
                transforms.RandShiftIntensityd(keys=['image'], offsets=0.05, prob=0.1,
                                               allow_missing_keys=True),
            ]

        if self.encode_latents == 'post':
            val_transforms += encode_latents_transforms()
            train_transforms += encode_latents_transforms()

        self.train_transform = train_transforms + [
            transforms.EnsureTyped(keys=['image', 'label', 'detail', 'vessel', 'spacing'], dtype=self.dtype, allow_missing_keys=True)
        ]
        self.val_transform = val_transforms + [
            transforms.EnsureTyped(keys=['image', 'label', 'detail', 'vessel', 'spacing'], dtype=self.dtype, allow_missing_keys=True)
        ]

        self.train_dataset = DatasetClass(data=self.train_data,
                                          transform=self.train_transform,
                                          **kwargs)
        self.val_dataset = DatasetClass(data=self.val_data,
                                        transform=self.val_transform,
                                        **kwargs)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            num_workers=self.num_workers
        )

    def construct_data(self, image_dir: str, label_dir: str, detail_dir: str):
        data = []
        keys = ['image', 'label', 'detail']

        def pair_data(data, dir, key):
            if dir:
                items = sorted(glob.glob(os.path.join(dir, '*')))
                if not data:
                    data = [{key: item, 'spacing': None} for item in items]
                else:
                    # make sure both images and items have same names
                    assert len(data) == len(items)
                    assert all(os.path.basename(o[k]).split('.')[0] == os.path.basename(item).split('.')[0]
                            for o, item in zip(data, items) for k in keys if k in o)
                    data = [{**d, key: item} for d, item in zip(data, items)]
            return data

        for i, key in enumerate(keys):
            data = pair_data(data, [image_dir, label_dir, detail_dir][i], key)

        return data
