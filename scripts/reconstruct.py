import os
import glob
import argparse
import torch

from util import write_image, gpu_vram_usage
from src.utils.constants import spec_extension
from src.data.utils import (
    SpacingResized,
    DimensionAwareOrientationd,
    SplitVesselLabelTransform
)
from src.processing.autoencode import VAEProcessor
from monai import transforms
from monai.utils.misc import ensure_tuple_rep


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input-dir', type=str, required=True,
                        help='Input directory containing images to reconstruct.')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory containing reconstructed images.')
    parser.add_argument('--label-dir', type=str, required=False,
                        help='directory containing annotations including vessels. For concat-mode VAE only.')
    parser.add_argument('--model', type=str, required=True,
                        help='AE Model checkpoint file.')
    parser.add_argument('--resolution', type=int, nargs='+', required=True,
                        help='Resolution size in all spatial dimension.')
    parser.add_argument('--spacing', type=float, nargs='*', default=None,
                        help='Resample to specific spacing in all spatial dimensions.')
    parser.add_argument('--copy-images', default=True, action=argparse.BooleanOptionalAction,
                        help='Whether to also copy the ground truth image to the output directory.')
    parser.add_argument('--gpu-summary', default=True, action=argparse.BooleanOptionalAction,
                        help='Print a short summary about GPU VRAM usage.')

    args, _ = parser.parse_known_args()
    return args


@torch.no_grad()
def main():
    args = parse_args()
    gpu_vram_used = []
    image_paths = sorted(glob.glob(os.path.join(args.input_dir, '*')))

    device = torch.device('cuda')

    VAEProcessor.init(args.model, device=device)

    assert image_paths, 'Input directory either does not exist or is empty.'
    SPECS = spec_extension(image_paths[0])

    label_paths = None
    if args.label_dir and VAEProcessor.concat_decode:
        label_paths = sorted(glob.glob(os.path.join(args.label_dir, '*')))
        assert all(os.path.basename(a).split('.')[0] == os.path.basename(b).split('.')[0] for a, b in zip(image_paths, label_paths))

    image_paths = [{args.type: img, 'path': img} for img in image_paths]

    if label_paths:
        image_paths = [{'label': lab, **o} for o, lab in zip(image_paths, label_paths)]

    if len(args.resolution) == 1:
        args.resolution = args.resolution[0]
    if args.spacing and len(args.spacing) == 1:
        args.spacing = args.spacing[0]

    resolution = ensure_tuple_rep(args.resolution, SPECS.dim)
    spacing = ensure_tuple_rep(args.spacing, SPECS.dim) if args.spacing else None

    tf = transforms.Compose([
        # Load images
        transforms.LoadImaged(keys=['image', 'label'], allow_missing_keys=True),
        # add channel dimension
        transforms.EnsureChannelFirstd(keys=['image', 'label'], allow_missing_keys=True),
        # init orientation, only applied to 3D
        DimensionAwareOrientationd(keys=['image', 'label'], dim=SPECS.dim, axcodes='RAS', allow_missing_keys=True),
        # spatially adapt spacing and resize input image and label
        SpacingResized(keys='image', resolution=resolution, spacing=spacing, mode='image',
                       pad_constant=SPECS.pad_constant_image, allow_missing_keys=True),
        SpacingResized(keys='image', resolution=resolution, spacing=spacing, mode='label',
                       pad_constant=SPECS.pad_constant_label, allow_missing_keys=True),
        SplitVesselLabelTransform(output_key='label', do_transform=label_paths is not None),
        # clip intensities and normalize
        transforms.ScaleIntensityRanged(keys=['image'], a_min=SPECS.clip_min, a_max=SPECS.clip_max,
                                        b_min=SPECS.a_min, b_max=SPECS.a_max, clip=True),
    ])

    z_mean, z_std = [], []

    for o in image_paths:
        o = tf(o)
        base_name = os.path.basename(o['path']).split('.')[0]
        out_dir = os.path.join(args.output_dir, base_name)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        affine = None if SPECS.dim == 2 else o[args.type].affine

        input = o[args.type].unsqueeze(0).to(device=device)
        concat = None
        if args.label_dir and VAEProcessor.concat_decode:
            concat = o['label'].unsqueeze(0).to(device=device)

        if args.gpu_summary:
            gpu_vram_used.append(gpu_vram_usage(device))

        out = VAEProcessor.encode(input)
        z_mean.append(out.mean().item())
        z_std.append(out.std().item())
        out = VAEProcessor.decode(out, concat)

        if args.gpu_summary:
                gpu_vram_used.append(gpu_vram_usage(device))

        output_path = os.path.join(out_dir, f'{base_name}.img-rec{SPECS.extension}')
        write_image(out[0], output_path, affine=affine)

        if args.copy_images:
            write_image(o['image'], os.path.join(out_dir, f'{base_name}.img{SPECS.extension}'), affine=affine)

    print('##### LATENT STATISTICS ###################################################')
    print(f'z mean: {sum(z_mean) / len(z_mean)}')
    print(f'z std: {sum(z_std) / len(z_std)}')
    print('#####################################################################')

    if args.gpu_summary:
        print('##### GPU SUMMARY ###################################################')
        print(f'Min GPU vRAM usage: {min(gpu_vram_used):.2f} GB')
        print(f'Max GPU vRAM usage: {max(gpu_vram_used):.2f} GB')
        print(f'Avg GPU vRAM usage: {(sum(gpu_vram_used) / len(gpu_vram_used)):.2f} GB')
        print('#####################################################################')


if __name__ == '__main__':
    main()
