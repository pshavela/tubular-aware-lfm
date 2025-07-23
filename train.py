import torch
import argparse
import warnings
import lightning as L

from src.autoencoder.vae import VQVAEModel
from src.processing.autoencode import VAEProcessor
from src.processing.generation import LGMProcessor
from src.data.medical import MedicalDataModule
from src.generative.diffusion.ddpm import LatentDiffusionModel
from src.generative.flow.flowmatcher import LatentFlowMatcherModel
from src.generative.hypernet.controlnet import SemanticControlNet
from src.utils.callbacks import GPUStatsCallback, ResumableEarlyStopping
from src.utils.ema_callback import EMA, EMAModelCheckpoint
from omegaconf import OmegaConf
from monai.config import print_config
from monai.utils import set_determinism
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.callbacks.stochastic_weight_avg import StochasticWeightAveraging


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=5,
                        help='Random seed to use.')
    parser.add_argument('--config', type=str, required=True,
                        help='Configuration file for model and data.')
    parser.add_argument('--precision', type=str, required=True,
                         help='Which floating point precision to use.')
    parser.add_argument('--gpus', default=None, type=int, choices=[None, 1, 2],
                        help='Number of GPUs to utilize. Can be at most 2. If 2 GPUs are specified, then '
                         + 'the first is used for training and second for preprocessing, otherwise all on one device.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train.')
    parser.add_argument('--stage', type=float, choices=[1, 2, 3], required=True,
                        help='Determines the stage to train.')
    parser.add_argument('--preprocess', default=False, action='store_true',
                        help='Whether to run pre-run the train/val dataloaders before training. Cleans up the '
                        + 'Processors afterwards, useful for PersistentDataset and limited GPU memory.')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to the checkpoint file. Will resume training state.')
    parser.add_argument('--gradient-clip', type=float, default=None,
                        help='Gradient clipping with specified norm.')
    parser.add_argument('--dim', type=int, choices=[2, 3], default=3,
                        help='Dimensions of dataset images')
    parser.add_argument('--latent-mode', type=str, choices=['diffusion', 'flow'], default='diffusion',
                        help='One of [diffusion, flow]. Which latent regression process to run.')
    parser.add_argument('--logger', type=str, choices=['wandb', 'tensorboard'],
                        help='One of [wandb, tensorboard]. Which logger to use.')
    parser.add_argument('--profiler', type=str, default=None,
                        help="Use Lightning's profiler. Use `--profiler simple` for simple mode.")

    args = parser.parse_args()
    return args


def init_callbacks(callbacks: dict, additional_gpu_devices = None):
    callbacks_ = []

    for callback in callbacks:
        if callback == 'earlystopping':
            callback = ResumableEarlyStopping(**callbacks[callback])
        elif callback == 'modelcheckpoint':
            if 'ema' in callbacks:
                callback = EMAModelCheckpoint(**callbacks[callback])
            else:
                callback = ModelCheckpoint(**callbacks[callback])
        elif callback == 'ema':
            callback = EMA(**callbacks[callback])
        elif callback == 'swa':
            callback = StochasticWeightAveraging(**callbacks[callback])
        elif callback == 'gpustats':
            callback = GPUStatsCallback(**callbacks[callback],
                                        additional_devices=additional_gpu_devices)

        callbacks_.append(callback)

    return callbacks_


def main(args):
    args = parse_args()

    set_determinism(seed=args.seed)
    print_config()
    # hide torch load warnings
    warnings.filterwarnings('ignore', '.*You are using `torch.load` with `weights_only=False` (the current default value)*')

    config = OmegaConf.load(args.config)

    # if two gpu devices are specified, then use first for preprocessing and second for training
    if args.gpus:
        train_device = 0
        prep_device = 'cuda:1' if args.gpus > 1 else 'cuda:0'
    else:
        train_device = 'cpu'
        prep_device = train_device

    dm = MedicalDataModule(dim=args.dim, device=prep_device, **config['data'])

    # for numerical stability in AMP mode
    config['model']['eps'] = 1e-6 if args.precision == '16-mixed' else 1e-8

    if args.stage == 1:
        config['model']['gradient_clip'] = args.gradient_clip
        model = VQVAEModel(**config['model'], logger_type=args.logger)
    elif args.stage == 2:
        model = LatentFlowMatcherModel if args.latent_mode == 'flow' else LatentDiffusionModel
        model = model(**config['model'], logger_type=args.logger)
    elif args.stage == 3:
        # init stage 2 model and freeze
        LGMProcessor.init(**config['processing']['generation'], type=args.latent_mode, device=train_device)
        model = SemanticControlNet(**config['model'], is_controlnet=True, logger_type=args.logger)

    if args.stage > 1:
        # init vae model and freeze
        VAEProcessor.init(**config['processing']['autoencode'], device=prep_device)

    additional_gpu_devices = [prep_device] if args.gpus and args.gpus > 1 else None
    callbacks = init_callbacks(config.get('callbacks', {}), additional_gpu_devices)

    largs = {
        'precision': args.precision,
        'log_every_n_steps': 5,
        'max_epochs': args.epochs,
        'callbacks': callbacks,
    }

    if args.gpus:
        largs['accelerator'] = 'gpu'
        largs['devices'] = [train_device]
    else:
        largs['accelerator'] = 'cpu'

    if args.logger == 'wandb':
        largs['logger'] = WandbLogger(log_model=True, config=config)
    else:
        largs['logger'] = True

    if args.profiler:
        largs['profiler'] = args.profiler

    # vqvae (stage 1) uses manual optimization, hence trainer cannot automatically clip gradients
    if args.gradient_clip and args.stage != 1:
        largs['gradient_clip_val'] = args.gradient_clip

    trainer = L.Trainer(**largs)
    dm.prepare_data(preprocess=args.preprocess)
    # clean up processors appropriately
    if args.preprocess:
        keep = 'decoder' if model.sample_image_every_n_epochs > 0 else None
        VAEProcessor.cleanup(keep)

    torch.set_float32_matmul_precision('high')
    trainer.fit(model, datamodule=dm, ckpt_path=args.resume)


if __name__ == '__main__':
    args = parse_args()
    main(args)
