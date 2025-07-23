import os
import re
import glob
import argparse
import torch
import time

from typing import Tuple, List, Dict

from util import write_image, gpu_vram_usage
from src.utils.constants import VOLUME, IMAGE
from src.processing.autoencode import VAEProcessor
from src.data.utils import (
    SpacingResized,
    PopulateSpacingTransform,
    DimensionAwareOrientationd,
    SplitVesselLabelTransform,
    EncodeVesselsTransformd,
    DilateBinaryLabelMapTransformd
)
from src.generative.diffusion.ddpm import LatentDiffusionModel
from src.generative.flow.flowmatcher import LatentFlowMatcherModel
from monai import transforms
from monai.data import Dataset, DataLoader
from monai.utils import set_determinism
from monai.utils.misc import ensure_tuple_rep
from omegaconf import OmegaConf


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, required=True,
                        help='YAML configuration file. Also used during training. If no configuration is specified,' +
                        'then unconditional model is assumed.')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory containing synthetic images.')
    parser.add_argument('--model', type=str, required=True,
                        help='Latent Generation Model checkpoint file path.')
    parser.add_argument('--latent-mode', type=str, choices=['flow', 'diffusion'], required=True,
                        help='Which latent generation model should be restored.')
    parser.add_argument('--num-samples', type=int, default=1,
                        help='How many samples to synthesize for each label.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed for reproducibility.')
    parser.add_argument('--amp', default=False, action=argparse.BooleanOptionalAction,
                        help='Run inference in automatic mixed precision mode.')

    args, _ = parser.parse_known_args()
    return args


def get_data_loader(SPECS, data: Dict):
    def construct_data(image_dir: str, label_dir: str, detail_dir: str):
        data = []
        keys = ['image', 'label', 'detail']

        def pair_data(data, dir, key):
            if dir:
                items = sorted(glob.glob(os.path.join(dir, '*')))
                if not data:
                    data = [{
                        key: item,
                        'spacing': None,
                        'base_name': re.search(r'\d+', os.path.basename(item))[0],
                    } for item in items]
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

    resolution = ensure_tuple_rep(data['resolution'], SPECS.dim)
    spacing = ensure_tuple_rep(data['spacing'], SPECS.dim)
    encode_vessels = data.get('encode_detail_pos', False)
    encode_vessels_padding_size = data.get('encode_detail_pos_size', None)
    encode_vessels_normalize = data.get('encode_detail_pos_normalize', True)
    encode_vessels_embedding_size = data.get('encode_detail_pos_emb_size', None)
    label_one_hot = data.get('label_one_hot', True)
    label_classes = data.get('number_classes', None)

    tf = [
        transforms.LoadImaged(
            keys=['image', 'label'],
            allow_missing_keys=True
        ),
        transforms.EnsureChannelFirstd(
            keys=['image', 'label'],
            allow_missing_keys=True
        ),
        DimensionAwareOrientationd(
            keys=['image', 'label'],
            dim=SPECS.dim,
            axcodes='RAS',
            allow_missing_keys=True
        ),
        SpacingResized(
            keys=['image'],
            resolution=resolution,
            spacing=spacing,
            mode='image',
            pad_constant=SPECS.pad_constant_image,
            allow_missing_keys=True
        ),
        SpacingResized(
            keys=['label'],
            resolution=resolution,
            spacing=spacing,
            mode='label',
            pad_constant=SPECS.pad_constant_label,
            allow_missing_keys=True,
        ),
        PopulateSpacingTransform(
            resolution=resolution,
            spacing=spacing,
            image_key='label',
        ),
        transforms.ScaleIntensityRanged(
            keys=['image'],
            a_min=SPECS.clip_min,
            a_max=SPECS.clip_max,
            b_min=SPECS.a_min,
            b_max=SPECS.a_max,
            clip=True,
            allow_missing_keys=True
        ),
    ]

    if encode_vessels:
        tf += [
            SplitVesselLabelTransform(do_transform=True),
            EncodeVesselsTransformd(
                keys=['detail'],
                pad_size=encode_vessels_padding_size,
                pos_enc_emb=encode_vessels_embedding_size,
                normalize_unit=encode_vessels_normalize,
                store_coords=True
            ),
        ]

    if label_one_hot:
        tf += [
            transforms.AsDiscreted(
                keys=['label'],
                to_onehot=label_classes
            )
        ]

    tf += [
        transforms.EnsureTyped(
            keys=['image', 'label', 'detail', 'vessel', 'spacing'],
            dtype=torch.float32,
            allow_missing_keys=True
        )
    ]

    data = construct_data(data.get('image_val_dir', ''), data.get('label_val_dir', ''), '')
    dataset = Dataset(data=data, transform=transforms.Compose(tf))
    dataloader = DataLoader(dataset, batch_size=1, num_workers=1)

    return dataloader


@torch.inference_mode()
def main():
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    GPU = []

    print(f'Running with AMP={args.amp}')
    if args.seed is not None:
        print(f"Fixing seed to {args.seed}")
        set_determinism(args.seed)

    config = OmegaConf.load(args.config)
    VAEProcessor.init(**config['processing']['autoencode'], device=device, load='decoder')

    if args.latent_mode == 'flow':
        model = LatentFlowMatcherModel.load_from_checkpoint(args.model, strict=False)
    else:
        model = LatentDiffusionModel.load_from_checkpoint(args.model, strict=False)

    model = model.to(device)
    model.freeze()

    dim = model.spatial_dims
    SPECS = VOLUME if dim == 3 else IMAGE

    # synthesize conditionally in case label validation directory was specified
    if 'label_val_dir' in config['data']:
        data_loader = get_data_loader(SPECS=SPECS, data=config['data'])

        for batch in data_loader:
            base_name = batch['base_name'][0]
            output_dir = os.path.join(args.output_dir, f'{base_name}-synth')

            if not os.path.exists(output_dir):
                os.mkdir(output_dir)

            affine = None if dim == 2 else batch['label'][0].affine

            with torch.no_grad(), torch.amp.autocast('cuda', enabled=args.amp):
                for n in range(1, args.num_samples + 1):
                    real = batch['image']
                    conditions = batch['label'].to(device)
                    spacing = batch['spacing'].to(device)
                    contexts = batch['detail'].to(device) if 'detail' in batch else None
                    context_coords = batch['detail_coords'].to(device) if 'detail_coords' in batch else None

                    concat = None
                    if VAEProcessor.concat_decode:
                        concat = conditions
                        if concat.shape[1] > 1:
                            concat = torch.argmax(concat, dim=1, keepdim=True)
                        concat = (concat == concat.max())
                        concat = DilateBinaryLabelMapTransformd(keys=['c'], iterations=2)({'c': concat[0].cpu()})['c']
                        concat = transforms.GaussianSmooth(sigma=0.75)(concat).unsqueeze(0).to(conditions)

                    start = time.time()
                    synth = model.sample_image(
                        conditions=conditions,
                        contexts=contexts,
                        context_coords=context_coords,
                        spacings=spacing,
                        concat=concat
                    )
                    delta = time.time() - start
                    print(f'[DELTA]: {delta} seconds')
                    GPU.append(gpu_vram_usage(device))

                    write_image(synth[0], os.path.join(output_dir, f'{n}-synth{SPECS.extension}'), affine=affine)
                    write_image(real[0], os.path.join(output_dir, f'{base_name}.img{SPECS.extension}'), affine=affine)
                    write_image(conditions[0], os.path.join(output_dir, f'{base_name}.label{SPECS.extension}'),
                                affine=affine, is_onehot=conditions.shape[1] > 1, denormalize_intensity=False)

                    torch.cuda.empty_cache()
    else:
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=args.amp):
            for n in range(1, args.num_samples + 1):
                if dim == 2:
                    spacing = [1.0, 1.0]
                    affine = None
                else:
                    affine = torch.eye(4)
                    affine[0][0] = spacing[0]
                    affine[1][1] = spacing[1]
                    affine[2][2] = spacing[2]
                    affine[-1, -1] = 1.0

                spacing = torch.FloatTensor(spacing).to(device).unsqueeze(0)
                start = time.time()
                synth = model.sample_image(conditions=None, spacings=spacing, contexts=None, context_coords=None)
                delta = time.time() - start
                print(f'[DELTA]: {delta} seconds')
                GPU.append(gpu_vram_usage(device))

                write_image(synth[0], os.path.join(args.output_dir, f'{n}-synth{SPECS.extension}'), affine=affine)

                torch.cuda.empty_cache()

    print('##### GPU SUMMARY ###################################################')
    print(f'Min GPU vRAM usage: {min(GPU):.2f} GB')
    print(f'Max GPU vRAM usage: {max(GPU):.2f} GB')
    print(f'Avg GPU vRAM usage: {(sum(GPU) / len(GPU)):.2f} GB')
    print('#####################################################################')


if __name__ == '__main__':
    main()
