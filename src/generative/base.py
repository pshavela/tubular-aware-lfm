from __future__ import annotations

import torch
import lightning as L

from src.utils.util import get_input, rand_int, cosine_warmup_lr_scheduler
from src.utils.log import LoggerWrapper
from src.processing.autoencode import VAEProcessor
from src.generative.unet import SemanticDiffusionModelUNet
from src.generative.conditioning import ConditioningEmbedding
from functools import partial
from typing import Tuple
from monai.utils.misc import ensure_tuple_rep
from lightning.pytorch.utilities import grad_norm
from generative.networks.nets.controlnet import ControlNet


class LatentGenerativeModel(L.LightningModule):
    """
    A generative model in the latent space, acting as a backbone for diffusion or conditional flow matching.
     Capable of conditioning on semantic label maps and contextual information such as spacing.

    Args:
        model_config: configuration for the velocity prediction model.
        spatial_dims: the spatial dimensions of the latent encodings (usually 3 for 3D, and 2 for 2D images).
        latent_resolution: the resolution of the latent encodings (equal size for height, width, (depth)).
        embedding_dim: the dimension of a codebook vector.
        learning_rate: learning rate. Adam with linear warmup scheduler and cosine annealing scheduler.
        learning_rate_min: The minimum learning rate reached by cosine annealing over the maximum number of steps.
        lr_warmup_steps: Step, up to which the learning rate linearly increases to the defined learning_rate and anneals
            after that.
        eps: Adam epsilon value.
        loss: Regression L_p loss type, default L_1.
        cfg: Classifier-free-guidance factor. Must be used in tandem with the DataModule label_drop_prob in src.data.medical.
        spade_cond: Can be one of three values, should be None when is_controlnet is set to True
            - None: no conditioning applied unless is_controlnet is set to True.
            - 'default': SPADE conditioning is applied with semantic map downscaled to match latent resolution.
            - Dictionary: SPADE conditioning is applied after the semantic label map is embedded with parameterized
                down-convolutions. See src.generative.conditioning for arguments. spatial_dims and in_channels must not
                be re-specified.
        num_classes: Number of semantic label classes. Only for SPADE or Controlnet conditioning.
        vessel_loss_weight: the weights by which the downsampled latent space binary vessel voxels will be weighted. Vessels
            should have max label class.
        z_mean: the mean of the raw latents. to be used for z-score normalization in combination with scaling_factor.
        scaling_factor: the scaling factor for the latents.
        is_controlnet: Whether this model is a ControlNet.
        controlnet_cond_scale: ControlNet scaling factor for residuals.
        inference_steps: Number of steps for the inference process
        latent_key: Each batch item is a dictionary. Determines the latent image key.
        label_key: key of the segmentation label map.
        original_image_key: key for the center slices of the original input image.
        original_label_key: key for the center slices of the original label map.
        logger_type: The type of the logger to use.
        sample_image_every_n_epochs: How often to sample images during validation.
        sample_image_process: Whether to also show intermediate process generation images.
        save_intermediate_steps: How many intermediate images should be saved during the sampling process for logging.
        checkpoint_path: Restore model weights by checkpoint.
    """
    def __init__(
        self,
        *,
        model_config: dict,
        spatial_dims: int,
        latent_resolution: int | Tuple[int, int] | Tuple[int, int, int],
        embedding_dim: int,
        learning_rate: float = 1e-4,
        learning_rate_min: float = 1e-6,
        lr_warmup_steps: float = 10000,
        eps: float = 1e-8,
        loss: str = 'l1',
        cfg: float = 1.0,
        spade_cond = None,
        num_classes: int = None,
        vessel_loss_weight: float = 0.0,
        z_mean: float = 0.0,
        scaling_factor: float = 1.0,
        is_controlnet: bool = False,
        controlnet_cond_scale: float = 1.0,
        inference_steps: int = 50,
        latent_key: str = 'image',
        label_key: str = 'label',
        original_image_key: str = 'original_image',
        original_label_key: str = 'original_label',
        spacing_key: str = 'spacing',
        context_key: str = 'detail',
        context_coords_key: str = 'detail_coords',
        vessel_key: str = 'vessel',
        logger_type: str = 'tensorboard',
        sample_image_every_n_epochs: int = 10,
        sample_image_process: bool = False,
        save_intermediate_steps: int = 5,
        checkpoint_path: str = None
    ):
        super().__init__()

        # transfer latent space info
        self.spatial_dims = spatial_dims
        self.embedding_dim = embedding_dim
        self.latent_resolution = ensure_tuple_rep(latent_resolution, self.spatial_dims)
        self.learning_rate = learning_rate
        self.learning_rate_min = learning_rate_min
        self.lr_warmup_steps = lr_warmup_steps
        self.eps = eps

        assert loss in ['l1', 'l2'], 'Flow Loss must be either l1 or l2.'
        self.loss = torch.nn.functional.mse_loss if (loss == 'l2') else torch.nn.functional.l1_loss
        self.cfg = cfg
        self.spade_cond = spade_cond
        self.num_classes = num_classes

        self.z_mean = z_mean
        self.scaling_factor = scaling_factor
        self.is_controlnet = is_controlnet
        self.controlnet_cond_scale = controlnet_cond_scale
        self.inference_steps = inference_steps
        self.latent_key = latent_key
        self.label_key = label_key
        self.original_image_key = original_image_key
        self.original_label_key = original_label_key
        self.spacing_key = spacing_key
        self.context_key = context_key
        self.context_coords_key = context_coords_key
        self.vessel_key = vessel_key
        self.logger_type = logger_type
        self.sample_image_every_n_epochs = sample_image_every_n_epochs
        self.sample_image_process = sample_image_process
        self.save_intermediate_steps = save_intermediate_steps
        self.vessel_loss_weight = vessel_loss_weight

        Model = SemanticDiffusionModelUNet if not self.is_controlnet else ControlNet
        dargs = {
            'spatial_dims' : spatial_dims,
            'in_channels': self.embedding_dim,
            'with_spacing_cond': True,
            'with_spade_cond': self.spade_cond is not None,
            **model_config,
        }
        if not self.is_controlnet:
            dargs['resolution'] = self.latent_resolution
            dargs['out_channels'] = self.embedding_dim

        # create paremeterized down-convolutional layers for semantic label map
        self.pre_spade = torch.nn.Identity()
        if self.spade_cond and self.spade_cond != 'default':
            self.pre_spade = ConditioningEmbedding(spatial_dims=self.spatial_dims,
                                                   in_channels=self.num_classes,
                                                   **self.spade_cond)
            dargs['label_nc'] = self.spade_cond['num_channels'][-1]
        elif self.spade_cond == 'default':
            dargs['label_nc'] = self.num_classes

        self.model = Model(**dargs)

        # load available checkpoint
        if checkpoint_path:
            self.restore_checkpoint(checkpoint_path)

        # store hparams
        self.save_hyperparameters()

    def loss_weights(self, vessels: torch.Tensor) -> torch.Tensor | None:
        if self.vessel_loss_weight > 0.0:
            weights = torch.ones(vessels.shape).to(vessels)
            weights[vessels == 1] = self.vessel_loss_weight
            return weights

        return None

    def _forward(
        self,
        z_1,
        t,
        condition = None,
        contexts = None,
        context_coords = None,
        spacings = None,
        down_residuals = None,
        mid_residual = None,
        vessels = None,
    ):
        """
        Performs a forward pass.

        Args:
            z_1: batch of latents of shape BERR[R], where E is self.embedding_dim and R is self.latent_resolution.
            t: Random time tensor.
            condition: Can be a semantic label map, ie. batch of one-hot encoded semantic label maps of shape BLHW[D],
                where L is the number of classes. It can also be any conditioning for the ControlNet module.
            contexts: contextual information processed with cross-attention of shape BN1.
            context_coords: positional information of contexts, optional.
            spacings: spacings tensor of shape B[spatial_dims].
            down_residuals: DownBlock residuals from the ControlNet to be injected to the noise predicition model.
            mid_residual: MidBlock residualas from the ControlNet to be injected to the noise predicition model.
            vessels: Downsampled binary (dilated) vessel mask for loss weight prioritization.

        Returns:
            loss value, z_1, down_residuals and mid_residual.
        """
        raise NotImplementedError

    def _rand_time(self, z_1):
        """
        Randomly samples time over a uniform interval. Discrete for the diffusion model, continuous for flow matching.

        Args:
            z_1: batch of latents of shape BERR[R], where E is self.embedding_dim and R is self.latent_resolution.

        Returns:
            Random time tensor.
        """
        raise NotImplementedError

    def forward(self, batch):
        z_1, labels, vessels, spacings, contexts, context_coords = get_input(
            batch,
            latent_key=self.latent_key,
            label_key=self.label_key,
            vessel_key=self.vessel_key,
            spacing_key=self.spacing_key,
            context_key=self.context_key,
            context_coords_key=self.context_coords_key,
        )

        # scale appropriately
        z_1 = (z_1 - self.z_mean) * self.scaling_factor
        t = self._rand_time(z_1)
        return self._forward(z_1=z_1, t=t,
                             condition=labels,
                             contexts=contexts,
                             context_coords=context_coords,
                             spacings=spacings,
                             vessels=vessels)

    def training_step(self, batch):
        loss, z_1, _, mid_residual = self.forward(batch)

        lr_scheduler = self.lr_schedulers()
        logs = {
            'train/loss': loss,
            'train/lr': lr_scheduler.get_last_lr()[0],
        }

        if mid_residual is not None:
            logs['train/mid_residual_mean'] = mid_residual.mean()
            logs['train/mid_residual_std'] = mid_residual.std()

        self.log_dict(logs)

        return loss

    def validation_step(self, batch):
        loss, z_1, _, mid_residual = self.forward(batch)

        logs = {
            'val/loss': loss,
        }

        if mid_residual is not None:
            logs['val/mid_residual_mean'] = mid_residual.mean()
            logs['val/mid_residual_std'] = mid_residual.std()

        self.log_dict(logs)

        return loss

    def on_before_optimizer_step(self, optimizer):
        # log mean gradient norm
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict({'mean_grad_norm': sum(norms.values()) / len(norms.values())})

    def optimizer_step(
        self,
        *args, **kwargs
    ):
        self.validate_gradients()
        return super().optimizer_step(*args, **kwargs)

    def validate_gradients(self):
        """
        https://github.com/Lightning-AI/pytorch-lightning/issues/4956#issuecomment-1722500885
        Skipping updates in case of unstable gradients
        https://github.com/Lightning-AI/lightning/issues/4956
        """
        valid_gradients = True
        for _, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any())
                if not valid_gradients:
                    break
        if not valid_gradients:
            print("Detected inf or nan values in gradients. Not updating model parameters")
            self.zero_grad()
            return False

        return True

    def on_validation_start(self):
        # init logger wrapper
        self.logger_wrapper = LoggerWrapper(self.logger, logger_type=self.logger_type)
        # init random validation batch number for logging random sampled validation batch image
        self.val_batch_idx = rand_int(0, self.trainer.num_val_batches[0])

    def on_validation_epoch_end(self):
        # update
        self.val_batch_idx = rand_int(0, self.trainer.num_val_batches[0])

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        slices_image, slices_label, labels, spacings, contexts, context_coords = get_input(
            batch,
            original_image_key=self.original_image_key,
            original_label_key=self.original_label_key,
            label_key=self.label_key,
            spacing_key=self.spacing_key,
            context_key=self.context_key,
            context_coords_key=self.context_coords_key,
        )

        controlnet = self.model if self.is_controlnet else None

        if (self.sample_image_every_n_epochs != -1 and self.global_step > 0
            and batch_idx == self.val_batch_idx and (self.current_epoch % self.sample_image_every_n_epochs) == 0):
            self.log_image(slices_image, slices_label, labels, contexts,
                           context_coords, spacings, controlnet=controlnet)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            # if pre_spade is nn.Identity then it wont have parameters
            list(self.model.parameters()) + list(self.pre_spade.parameters()),
            lr=self.learning_rate,
            eps=self.eps,
        )

        lr_scheduler = cosine_warmup_lr_scheduler(
            optimizer=optimizer,
            max_steps=self.trainer.estimated_stepping_batches,
            warmup_steps=self.lr_warmup_steps,
            lr=self.learning_rate,
            min_lr=self.learning_rate_min
        )

        # Perform LR scheduling after every step
        return {
            'optimizer' : optimizer,
            'lr_scheduler' : {
                'scheduler': lr_scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }

    def restore_checkpoint(self, checkpoint_path: str):
        state_dict = torch.load(checkpoint_path)["state_dict"]
        self.load_state_dict(state_dict)

    def _sample_latents(
        self,
        z_0,
        conditions,
        contexts,
        context_coords,
        spacings,
        controlnet = None,
        cond_scale: float = 1.0,
        save_intermediates: bool = False
    ):
        """
        Sample latent representations of images, potentially conditioned on semantic labels.

        Args:
            z_0: The initial prior starting values for each latent in the batch.
            conditions: The segmentation labels. Can be None.
            contexts: Additional contextual information.
            context_coords: Positional information for context tokens.
            spacings: spacing information.
            controlnet: ControlNet forward function for residual injection.
            cond_scale: Conditioning scale value, for ControlNet residuals.
            save_intermediates: whether to save the the intermediates along the flow path.

        Returns:
            Either list of intermediates and final sample, or final sample of shapes BERR[R].
        """
        raise NotImplementedError

    def sample_image(
        self,
        conditions,
        contexts,
        context_coords,
        spacings,
        controlnet = None,
        cond_scale: float = 1.0,
        save_intermediates: bool = False,
        concat: torch.Tensor = None,
    ):
        """
        Sample images unconditionally or conditioned on semantic labels.

        Args:
            conditions: semantic labels of shape BLHW[D], where L is number of one-hot encoding classes.
            contexts: additional contextual information, shape BN, where N is dimension of
                contextual information.

            spacings: spacing information.
            controlnet: ControlNet forward function. Can be used to inject residuals for conditioning.
            cond_scale: Conditioning scale value, for ControlNet residuals.
            save_intermediates: whether to also save the intermediates during sampling.
            concat: if the VQVAE decoder requires a concatenation with an additional tensor, this
                will not be None.

        Returns:
            Either list of intermediates and final sample, or final sample of shapes BCHW[D].
        """
        # BERR[R], where R is latent resolution size
        size = (spacings.shape[0], self.embedding_dim,) + self.latent_resolution

        # use full precision
        z_0 = torch.randn(size, device=self.device).float()
        z = self._sample_latents(z_0, conditions, contexts, context_coords, spacings, controlnet, cond_scale, save_intermediates)

        # in case vessels are channel-wise concatted to each decoder block
        if VAEProcessor.concat_decode and concat is None:
            concat = torch.argmax(conditions, dim=1, keepdim=True)
            concat = (concat == concat.max()).to(conditions)

        if save_intermediates:
            y_1 = [VAEProcessor.decode(self.z_mean + (z_ / self.scaling_factor), concat) for z_ in z]
        else:
            y_1 = VAEProcessor.decode(self.z_mean + (z / self.scaling_factor), concat)

        return y_1

    def log_image(
        self,
        slices_image: torch.Tensor,
        slices_label: torch.Tensor,
        conditions: torch.Tensor,
        contexts: torch.Tensor | None,
        context_coords: torch.Tensor | None,
        spacings: torch.Tensor,
        controlnet = None,
    ):
        slices_image = slices_image[0].float().cpu().numpy()

        sample_image_cond = partial(self.sample_image,
                                    conditions=conditions,
                                    contexts=contexts,
                                    context_coords=context_coords,
                                    spacings=spacings,
                                    controlnet=controlnet,
                                    cond_scale=self.controlnet_cond_scale)

        if self.sample_image_process:
            synths = sample_image_cond(save_intermediates=True)
            # single image batch
            synths = [s[0].float().cpu().numpy() for s in synths]
            synth = synths[-1]
            # log sampling process
            self.logger_wrapper.log_sampling_process(synths, step=self.trainer.global_step)
        else:
            synth = sample_image_cond(save_intermediates=False)
            # single image batch
            synth = synth[0].float().cpu().numpy()

        if controlnet or self.spade_cond:
            # upscale labels in case it was downsampled during data preprocessing pipeline
            slices_label = slices_label[0].float().cpu().numpy()
            self.logger_wrapper.log_conditional_image(slices_image, synth, slices_label, prefix='conditional',
                                                      caption='Top: Ground Truth, Middle: Semantic Map, Bottom: Synthetic',
                                                      is_slices=self.spatial_dims > 2, step=self.trainer.global_step)
        else:
            self.logger_wrapper.log_image_pair(slices_image, synth, prefix='unconditional',
                                               caption='Top: Ground Truth, Bottom: Synthetic',
                                               full_real=False, step=self.trainer.global_step)
