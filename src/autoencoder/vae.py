import torch
import lightning as L

from src.autoencoder.loss import VQGANLoss
from src.utils.log import LoggerWrapper
from src.utils.util import get_input, rand_int, cosine_warmup_lr_scheduler
from src.autoencoder.vqvae import VQVAE as VQVAEModule


class VQVAEModel(L.LightningModule):
    """
    The VQVAE model inspired by the Latent Diffusion Model https://arxiv.org/abs/2112.10752 and
    Morphology-preserving Autoregressive 3D Generative Modelling of the Brain by Tudosiu et al.
    (https://arxiv.org/pdf/2209.03177.pdf).

    Args:
        vae_config: configuration for the MONAI Generative VQVAE model.
        loss_config: configuration for the loss.
        learning_rate: learning rate of the discriminator. Adam with exponential decay of gamma 0.99999.
        learning_rate_generator_scale: The scale with which the learning_rate will be scaled by and used as the
            learning rate for the generator. Adam with exponential decay of gamma 0.99999.
        eps: Epsilon value for Adam optimizer.
        concat_vessels: whether to concatenate channel-wise the single-channel downsampled vessel maps to the
            decoder blocks.
        gradient_clip: Norm clipping value for gradients. Since manually done need to specify in config file.
        image_key: Each batch item is a dictionary containing images and possibly labels. Determines the image key.
        logger_type: The type of the logger to use.
        log_image_every_n_epochs: How often per epoch to store reconstruction images.
        checkpoint_path: Restore model weights by checkpoint.
    """
    def __init__(
        self,
        *,
        spatial_dims: int,
        vae_config: dict,
        loss_config: dict = {},
        learning_rate: float = 1e-4,
        learning_rate_min: float = 1e-6,
        lr_warmup_steps: int = 10000,
        learning_rate_generator_scale: float = 0.2,
        eps: float = 1e-8,
        concat_vessels: bool = False,
        gradient_clip: float = None,
        image_key: str = 'image',
        vessel_key: str = 'detail',
        logger_type: str = 'tensorboard',
        log_image_every_n_epochs: int = 10,
        checkpoint_path: str = None
    ):
        super().__init__()

        # transform configuration
        compression = len(vae_config['num_channels'])
        vae_config['downsample_parameters'] = compression * [(2, 4, 1, 1)]
        vae_config['upsample_parameters'] = compression * [(2, 4, 1, 1, 0)]

        self.vqvae = VQVAEModule(spatial_dims=spatial_dims, **vae_config,
                                 concat_decoder=int(concat_vessels))
        if loss_config:
            self.vqloss = VQGANLoss(spatial_dims=spatial_dims, **loss_config)

        self.image_key = image_key
        self.vessel_key = vessel_key
        self.learning_rate = learning_rate
        self.learning_rate_min = learning_rate_min
        self.lr_warmup_steps = lr_warmup_steps
        self.learning_rate_generator_scale = learning_rate_generator_scale

        self.eps = eps
        self.concat_vessels = concat_vessels
        self.gradient_clip = gradient_clip
        self.logger_type = logger_type
        self.log_image_every_n_epochs = log_image_every_n_epochs

        # load available checkpoint
        if checkpoint_path:
            self.restore_checkpoint(checkpoint_path)

        # need to manually optimize since we have multiple optimizers
        self.automatic_optimization = False

        # save hyperparameters
        self.save_hyperparameters()

    def decoder_last_layer_weights(self):
        # last layer of the convolutional block
        return self.vqvae.decoder.blocks[-1][-1].weight

    def encode_stage_2_inputs(
            self,
            x: torch.Tensor,
            quantized: bool = False,
            normalize: bool = True,
            vmin: float = -1.0,
            vmax: float = 1.0
        ):
        """
        Encodes the input for the latent generative model.

        Args:
            x: input of shape BCHW[D].
            quantized: whether to quantize the encoded input.
            normalize: Whether to normalize the encoded input to [vmin, vmax].
            vmin: minimum normalization bound.
            vmax: maximum normalization bound.

        Returns:
            encoded input of shape BEH'W'[D'], where E is the embedding dimension.
        """
        enc = self.vqvae.encode_stage_2_inputs(x, quantized)

        if normalize:
            cmin = self.vqvae.quantizer.quantizer.embedding.weight.min()
            cmax = self.vqvae.quantizer.quantizer.embedding.weight.max()
            enc = (vmax - vmin) * ((enc - cmin) / (cmax - cmin)) + vmin

        return enc

    def decode_stage_2_outputs(
            self,
            l: torch.Tensor,
            quantize: bool = True,
            denormalize: bool = True,
            concat: torch.Tensor = None,
            vmin: float = -1.0,
            vmax: float = 1.0
        ):
        """
        Decodes the encoded latents for the latent generative model.

        Args:
            l: latents of shape BEH'W'[D'], where E is the embedding dimension..
            quantize: whether to quantize the latents.
            denormalize: Whether to denormalize the encoded latents from [vmin, vmax] to the raw encoded space.
            vmin: minimum normalization bound.
            vmax: maximum normalization bound.

        Returns:
            decoded latents of shape BCHW[D].
        """
        if denormalize:
            cmin = self.vqvae.quantizer.quantizer.embedding.weight.min()
            cmax = self.vqvae.quantizer.quantizer.embedding.weight.max()
            l = (cmax - cmin) * ((l - vmin) / (vmax - vmin)) + cmin

        if quantize:
            return self.vqvae.decode_stage_2_outputs(l, concat)

        return self.vqvae.decode(l, concat)

    def forward(self, x, vessel_mask: bool = None, with_latents: bool = False):
        z_raw = self.vqvae.encode(x)
        z_quant, quantization_losses = self.vqvae.quantize(z_raw)
        reconstruction = self.vqvae.decode(z_quant, vessel_mask if self.concat_vessels else None)

        if not with_latents:
            return reconstruction, quantization_losses

        return reconstruction, quantization_losses, z_raw, z_quant

    def training_step(self, batch):
        opt_gen, opt_disc = self.optimizers()

        x, vessel_mask = get_input(batch, image_key=self.image_key, vessel_key=self.vessel_key)
        reconstruction, quantization_loss = self.forward(x, vessel_mask)

        def do_step(opt):
            if self.gradient_clip:
                self.clip_gradients(opt, gradient_clip_val=self.gradient_clip, gradient_clip_algorithm='norm')
            opt.step()

        ###############################################################################################################
        # first optimize over generator (VQVAE)
        self.toggle_optimizer(opt_gen)

        loss_generator, log_gen = self.vqloss.forward(x, reconstruction,
                                                      quantization_loss,
                                                      self.decoder_last_layer_weights(),
                                                      vessel_mask,
                                                      self.global_step,
                                                      optimize_over='generator',
                                                      split='train')

        self.manual_backward(loss_generator)
        do_step(opt_gen)
        opt_gen.zero_grad()
        self.untoggle_optimizer(opt_gen)
        ###############################################################################################################

        ###############################################################################################################
        # optimize over discriminator
        self.toggle_optimizer(opt_disc)

        loss_discriminator, log_disc = self.vqloss.forward(x, reconstruction,
                                                           quantization_loss,
                                                           None,
                                                           None,
                                                           self.global_step,
                                                           optimize_over='discriminator',
                                                           split='train')

        self.manual_backward(loss_discriminator)
        do_step(opt_disc)
        opt_disc.zero_grad()
        self.untoggle_optimizer(opt_disc)
        ###############################################################################################################

        # schedulers
        lr_scheduler_gen, lr_scheduler_disc = self.lr_schedulers()
        lr_scheduler_gen.step()
        # only when adverserial loss has started
        if torch.is_nonzero(loss_discriminator):
            lr_scheduler_disc.step()

        self.log_dict({'train/lr_generator': lr_scheduler_gen.get_last_lr()[0],
                       'train/lr_discriminator': lr_scheduler_disc.get_last_lr()[0],
                       **log_gen, **log_disc})

    def validation_step(self, batch):
        x, vessel_mask = get_input(batch, image_key=self.image_key, vessel_key=self.vessel_key)
        reconstruction, quantization_loss, z_raw, z_quant = self.forward(x, vessel_mask, with_latents=True)

        _, log_gen = self.vqloss.forward(x, reconstruction,
                                         quantization_loss,
                                         self.decoder_last_layer_weights(),
                                         vessel_mask,
                                         self.global_step,
                                         optimize_over='generator',
                                         split='val')

        _, log_disc = self.vqloss.forward(x, reconstruction,
                                          quantization_loss,
                                          None,
                                          None,
                                          self.global_step,
                                          optimize_over='discriminator',
                                          split='val')

        self.log_dict({
            'val/z_mean': z_raw.mean(),
            'val/z_std': z_raw.std(),
            'val/q_mean': z_quant.mean(),
            'val/q_std': z_quant.std(),
            **log_gen,
            **log_disc
        })

        # need for on_validation_batch_end()
        return x, reconstruction

    def on_validation_start(self):
        # init logger wrapper
        self.logger_wrapper = LoggerWrapper(self.logger, logger_type=self.logger_type)
        # init random validation batch number for logging random reconstructed validation batch image
        self.val_batch_idx = rand_int(0, self.trainer.num_val_batches[0])

    def on_validation_epoch_end(self):
        # update
        self.val_batch_idx = rand_int(0, self.trainer.num_val_batches[0])

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        x, reconstruction = outputs
        if batch_idx == self.val_batch_idx and (self.current_epoch % self.log_image_every_n_epochs) == 0 and self.log_image_every_n_epochs > 0:
            self.log_image(x, reconstruction)

            # randomly also log codebook vectors usage for debugging
            encoding_indices = self.vqvae.index_quantize(x)
            self.log_dict({
                'val/rand_image_codebook_usage': len(encoding_indices.detach().flatten().unique())
            })

    def configure_optimizers(self):
        lr_disc = self.learning_rate
        lr_gen = self.learning_rate_generator_scale * self.learning_rate

        opt_gen = torch.optim.Adam(self.vqvae.parameters(),
                                   lr=lr_gen, eps=self.eps)
        opt_disc = torch.optim.Adam(self.vqloss.discriminator.parameters(),
                                    lr=lr_disc, eps=self.eps)

        lr_scheduler_gen = cosine_warmup_lr_scheduler(
            optimizer=opt_gen,
            max_steps=self.trainer.estimated_stepping_batches,
            warmup_steps=self.lr_warmup_steps,
            lr=lr_gen,
            min_lr=self.learning_rate_generator_scale * self.learning_rate_min
        )

        lr_scheduler_disc = cosine_warmup_lr_scheduler(
            optimizer=opt_disc,
            max_steps=self.trainer.estimated_stepping_batches,
            warmup_steps=self.lr_warmup_steps,
            lr=lr_disc,
            min_lr=self.learning_rate_min
        )

        return [opt_gen, opt_disc], [lr_scheduler_gen, lr_scheduler_disc]

    def restore_checkpoint(self, checkpoint_path: str):
        state_dict = torch.load(checkpoint_path)["state_dict"]
        self.load_state_dict(state_dict)

    def log_image(self, input: torch.Tensor, reconstruction: torch.Tensor):
        """
        Log the middle slice of the first image of a specific batch for each the original input and the reconstructed
        volume.

        Args:
            input: The batch input of shape B1HW(D).
            reconstruction: The batch of reconstructions of shape B1HW(D).
        """

        # take first image each
        input = input[0].float().cpu().numpy()
        reconstruction = reconstruction[0].float().cpu().numpy()
        self.logger_wrapper.log_image_pair(input, reconstruction,
                                           prefix='reconstruction', caption='Top: Ground Truth, Bottom: Reconstruction',
                                           step=self.trainer.global_step)
