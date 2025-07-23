import os
import glob
import argparse
import torch

from util import spec_extension, gpu_vram_usage
from src.data.utils import DimensionAwareOrientationd
from monai import transforms
from monai.utils.misc import first
from monai.data import Dataset, DataLoader
from monai.metrics.fid import FIDMetric
from monai.metrics.regression import SSIMMetric, MultiScaleSSIMMetric, PSNRMetric
from generative.losses.perceptual import PerceptualLoss


GPU_VRAM_USAGE = []


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--reconstruction-dir', type=str, required=False,
                        help='Root directory containing subfolders with reconstruction and real images.')
    parser.add_argument('--label-dir', type=str, required=False,
                        help='Root directory containing semantic label maps for vesselness response metric. ' +
                        'Will extract the highest label class as vessels automatically.')
    parser.add_argument('--synthesis-dir', type=str, required=False,
                        help='Root directory containing subfolders with synthesized and real images.')
    parser.add_argument('--num-workers', type=int, required=False, default=0,
                        help='Number of worker threads for the data loading process.')
    parser.add_argument('--gpu-summary', default=True, action=argparse.BooleanOptionalAction,
                        help='Print a short summary about GPU VRAM usage.')

    args, _ = parser.parse_known_args()
    return args


def image_id(path: str):
    b = os.path.basename(path)
    return b[:b.index('.')]


def track_gpu_usage(device: torch.device):
    global GPU_VRAM_USAGE
    if device.type != 'cpu':
        GPU_VRAM_USAGE.append(gpu_vram_usage(device))


@torch.no_grad()
def compute_metrics_vae(
    parent_dir: str,
    kernel_size: int = 4,
    num_workers: int = 0,
    device = None,
):
    global GPU_VRAM_USAGE
    print('Computing metrics for reconstructed images...')
    paths = sorted(glob.glob(os.path.join(parent_dir, '*')))
    paths = [p for p in paths if os.path.isdir(p)]
    _ = sorted(glob.glob(os.path.join(paths[0], '*')))
    SPECS = spec_extension(_[0])
    ext = SPECS.extension
    dim = SPECS.dim

    real_images = [glob.glob(os.path.join(p, f'*.img{ext}'))[0] for p in paths]
    reconstructed_images = [glob.glob(os.path.join(p, f'*.img-rec{ext}'))[0] for p in paths]

    lpips = PerceptualLoss(spatial_dims=dim).to(device)
    ssim = SSIMMetric(spatial_dims=dim, data_range=1.0, win_size=kernel_size)
    ms_ssim = MultiScaleSSIMMetric(spatial_dims=dim, data_range=1.0, kernel_size=kernel_size)
    psnr = PSNRMetric(max_val=1.0)

    scores = {
        'ssim': [],
        'msssim' : [],
        'lpips': [],
        'psnr' : [],
    }

    def get_data_loader(real_images, reconstructed_images, dim: int):
        lower = -1000 if (dim == 3) else 0
        upper = 1000 if (dim == 3) else 255

        data = [{'real': real_img, 'fake': fake_img} for real_img, fake_img in zip(real_images, reconstructed_images)]
        assert all(image_id(o['real']) == image_id(o['fake']) for o in data)
        keys = list(first(data).keys())

        tf = transforms.Compose([
            transforms.LoadImaged(keys=keys),
            transforms.EnsureChannelFirstd(keys=keys),
            DimensionAwareOrientationd(keys=keys, dim=dim, axcodes='RAS'),
            transforms.ScaleIntensityRanged(keys=['real', 'fake'], a_min=lower, a_max=upper, b_min=0.0, b_max=1.0, clip=True),
        ])

        ds = Dataset(data=data, transform=tf)
        data_loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers)
        return data_loader

    data_loader = get_data_loader(real_images, reconstructed_images, dim=dim)

    for i, o in enumerate(data_loader):
        if (i % 5) == 0:
            print('Computing similarity scores for batch', i)
        real: torch.Tensor = o['real'].to(device)
        fake: torch.Tensor = o['fake'].to(device)
        track_gpu_usage(device)

        scores['ssim'].append(ssim(fake, real))
        track_gpu_usage(device)
        scores['msssim'].append(ms_ssim(fake, real))
        track_gpu_usage(device)
        scores['lpips'].append(lpips(fake, real))
        track_gpu_usage(device)
        scores['psnr'].append(psnr(fake, real))

    return {score: torch.stack(values).squeeze() for score, values in scores.items()}


@torch.no_grad()
def compute_metrics_lgm(
    parent_dir: str,
    num_workers: int = 0,
    device = None,
):
    global GPU_VRAM_USAGE
    print('Computing LGM scores for generated images...')
    paths = sorted(glob.glob(os.path.join(parent_dir, '*')))
    paths = [p for p in paths if os.path.isdir(p)]
    _ = sorted(glob.glob(os.path.join(paths[0], '*')))
    SPECS = spec_extension(_[0])
    ext = SPECS.extension
    dim = SPECS.dim

    real_images = [{'image': glob.glob(os.path.join(p, f'*.img{ext}'))[0]} for p in paths]
    synth_images = [{'image': s} for p in paths for s in glob.glob(os.path.join(p, f'*synth{ext}'))]

    fid = FIDMetric()

    # same as MAISI: https://github.com/Project-MONAI/MONAI/discussions/8243#discussioncomment-11430253
    feature_extractor = torch.hub.load('Warvito/radimagenet-models',
                                       model='radimagenet_resnet50', verbose=True, trust_repo=True).to(device)
    feature_extractor.eval()

    track_gpu_usage(device)

    def get_data_loader(data):
        keys = list(first(data).keys())

        tf = transforms.Compose([
            transforms.LoadImaged(keys=keys),
            transforms.EnsureChannelFirstd(keys=keys),
            DimensionAwareOrientationd(keys=keys, dim=dim, axcodes='RAS'),
            transforms.ScaleIntensityRanged(keys=keys, a_min=SPECS.clip_min, a_max=SPECS.clip_max,
                                            b_min=SPECS.a_min, b_max=SPECS.a_max)
        ])

        ds = Dataset(data=data, transform=tf)
        data_loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers)
        return data_loader

    real_data_loader = get_data_loader(real_images)
    synth_data_loader = get_data_loader(synth_images)

    def compute_features(data_loader, dim: int):
        features = [[], [], []]
        for i, o in enumerate(data_loader):
            if (i % 5) == 0:
                print('Extracting features for image batch', i)
            img = o['image'].to(device)
            track_gpu_usage(device)

            if dim == 2:
                # img is 11HW
                # resnet expects 3 channels, so just duplicate
                img = img.repeat(1, 3, 1, 1)
                out = feature_extractor(img)
                # spatial average
                out = out.mean([2, 3], keepdim=False)
                features[0].append(out)
            else:
                # for each plane
                for d in [1, 2, 3]:
                    # img is 1CHWD
                    imgs = torch.unbind(img, dim=d + 1)
                    # imgs is BCHW
                    imgs = torch.vstack(imgs)
                    # resnet expects 3 channels, so just duplicate
                    imgs = imgs.repeat(1, 3, 1, 1)
                    out = feature_extractor(imgs)
                    # spatial average
                    out = out.mean([2, 3], keepdim=False)
                    features[d - 1].append(out)

            track_gpu_usage(device)

        return features

    print('Extracting real image features...')
    real_features = compute_features(real_data_loader, dim=dim)
    print('Extracting synth image features...')
    synth_features = compute_features(synth_data_loader, dim=dim)
    dims = [1] if dim == 2 else [1, 2, 3]
    fids = []
    for d in dims:
        rf = torch.vstack(real_features[d - 1])
        sf = torch.vstack(synth_features[d - 1])
        fids.append(fid(sf, rf).item())

    out = {'fid': fids}

    return out

def vae_summary(scores_vae) -> str:
    summary = '##### VAE SUMMARY #############################################################\n'
    pad = max(len(s) for s in scores_vae.keys()) + 1
    for score, values in scores_vae.items():
        summary += f'{score.upper().ljust(pad)} Min: {values.min():.8f}, Max: {values.max():.8f} , Mean: {values.mean():.8f}(+-{values.std():.8f})\n'
    summary += '##############################################################################\n'
    return summary


def lgm_summary(fids_lgm) -> str:
    summary = '##### LGM SUMMARY #############################################################\n'
    summary += f'FID: ' +  ', '.join(f'[axis {i + 1}] {fid:.8f}' for i, fid in enumerate(fids_lgm['fid'])) + '\n'
    if 'vr' in fids_lgm:
        values = fids_lgm['vr']
        summary += f'VR Min: {values.min():.8f}, Max: {values.max():.8f} , Mean: {values.mean():.8f}(+-{values.std():.8f})\n'
    summary += '###############################################################################\n'
    return summary


def main():
    global GPU_VRAM_USAGE
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.reconstruction_dir:
        scores_vae = compute_metrics_vae(args.reconstruction_dir, num_workers=args.num_workers, device=device)
        print(vae_summary(scores_vae))

    if args.synthesis_dir:
        fids_lgm = compute_metrics_lgm(args.synthesis_dir, num_workers=args.num_workers, device=device)
        print(lgm_summary(fids_lgm))

    if args.gpu_summary:
        print('##### GPU SUMMARY ###################################################')
        print(f'Min GPU vRAM usage: {min(GPU_VRAM_USAGE):.2f} GB')
        print(f'Max GPU vRAM usage: {max(GPU_VRAM_USAGE):.2f} GB')
        print(f'Avg GPU vRAM usage: {(sum(GPU_VRAM_USAGE) / len(GPU_VRAM_USAGE)):.2f} GB')
        print('#####################################################################')


if __name__ == '__main__':
    main()
