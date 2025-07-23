# Tubular Anatomy-Aware 3D Semantically Conditioned Image Synthesis

Code for the paper
[Janluka Janelidze, Lukas Folle, Nassir Navab, and Mohammad Farid Azampour: Tubular Anatomy-Aware 3D Semantically Conditioned Image Synthesis, 2025](https://todo.com).

# Training
It is recommended to run it on a single GPU, multi-GPU not tested yet.

## Dataset
The coronary CTA dataset is from [Zeng et. al.: ImageCAS: A Large-Scale Dataset and Benchmark for Coronary Artery Segmentation based on Computed Tomography Angiography Images](https://github.com/XiaoweiXu/ImageCAS-A-Large-Scale-Dataset-and-Benchmark-for-Coronary-Artery-Segmentation-based-on-CT).


## VQVAE
```sh
# Randomly cropped subvolumes
python3 train.py --seed 5 --stage 1 --dim 3 --config configs/stage_1_3d_f4.yml --epochs 100 --gpus 1 --precision 16-mixed --logger tensorboard

# Full volume finetuning, change configuration yml and set checkpoint path
python3 train.py --seed 5 --stage 1 --dim 3  --config configs/stage_1_3d_f4.yml --epochs 300 --gpus 1 --precision 16-mixed --logger tensorboard --resume <path/to/last.ckpt>
```

#### Configuration File Notes
- `dataset_type: "persistent"` provides the best performance, `"cache"` is equally as performant but caused stability issues on some cluster instances. Additionally, the persistent data set will create a directory *persistent_cache* with the preprocessed tensors, which can be reused for multiple runs.
- **Since the persistent cache remains, in case of minor changes to the data configuration in the config file, the persistent cache must be deleted, otherwise the new changes will not be applied**


## Latent Flow Matching Model
```sh
python3 train.py --seed 5 --stage 2 --dim 3 --config config/stage_2_3d_f4_lfm_base.yml --epochs 500 --gpus 1 --latent-mode flow --precision 16-mixed --logger tensorboard
```

#### Configuration File Notes
- Since encoding the images to latents is done on the specified GPU device (CPU is heavily discouraged), parallelization is not possible and
therefore `num_workers: 0` should be used (See. [MONAI ToDevice](https://docs.monai.io/en/stable/transforms.html#monai.transforms.ToDevice)).
**However**, once the persistent dataset has been fully created,
`num_workers` can be increased in the next run.


# Inference

## Reconstruction
Reconstruction with the pretrained VQ-VAE can be performed with the [reconstruct.py](./scripts/reconstruct.py) script.

```sh
PYTHONPATH=./:$PYTHONPATH python3 scripts/reconstruct.py --input-dir <in>  --output-dir <out> --model <path/to/model/ckpt> --resolution 256 256 128 --spacing 0.7 0.7 1.05
```

## Synthesis
Synthetic images with the pretrained VQ-VAE and latent model can be generated with the [synthesize.py](./scripts/synthesize.py) script.

```sh
PYTHONPATH=./:$PYTHONPATH python3 scripts/synthesize.py --seed 0 --output-dir <out> --model <path/to/model/ckpt> --latent-mode flow --config <path/to/train/config/yml>
```


# Evaluation
The script [evaluations.py](./scripts/evaluations.py) can be used for the VAE and the LFM model to compute relevant metrics.

# Debugging

## Troubleshooting
If logging is done with _Weights and Biases_, during debugging its best to disable wandb by running `wandb offline` in the repo directory before debugging.
