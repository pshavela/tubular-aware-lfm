import torch

from src.generative.base import LatentGenerativeModel
from src.generative.diffusion.scheduler import DDPMScheduler
from functools import partial


class LatentDiffusionModel(LatentGenerativeModel):
    """
    A diffusion model in the latent space with/without SPADE conditioning.

    Args:
        train_steps: Number of time steps for training diffusion process.
        inference_steps: Number of diffusion steps during sampling process.
        save_intermediate_steps: How many intermediate images should be saved during the sampling process for logging.
        schedule: Which noise schedule to use.
        clip_sample: Whether to clip noise samples during process to [-1, 1].
        kwargs: Arguments for the base class.
    """
    def __init__(self,
                 *,
                 train_steps: int = 1000,
                 inference_steps: int = 1000,
                 save_intermediate_steps: int = 100,
                 schedule: str = 'scaled_linear_beta',
                 clip_sample: bool = False,
                 **kwargs):
        super().__init__(inference_steps=inference_steps,
                         save_intermediate_steps=save_intermediate_steps,
                         **kwargs)
        self.train_steps = train_steps

        if schedule == 'scaled_linear_beta':
            self.scheduler = DDPMScheduler(train_steps, schedule='scaled_linear_beta',
                                           beta_start=0.0015, beta_end=0.0195,
                                           clip_sample=clip_sample)
            self.scheduler.set_timesteps(inference_steps, 'leading')
        elif schedule == 'cosine_poly':
            # We follow the paper https://arxiv.org/pdf/2305.08891 for a custom cosine and timesteps implementation
            # with velocity prediction
            self.scheduler = DDPMScheduler(train_steps, schedule='cosine_poly',
                                           prediction_type='v_prediction',
                                           clip_sample=clip_sample)
            self.scheduler.set_timesteps(inference_steps, 'linspace')

    def _rand_time(self, z_1):
        t = torch.randint(0, self.train_steps, (z_1.shape[0],), device=self.device).long()
        return t

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
        # sample normal noise
        noise = torch.randn_like(z_1).to(z_1)
        noisy_z = self.scheduler.add_noise(original_samples=z_1, noise=noise, timesteps=t)

        predict_noise = partial(self.model.forward,
                                x=noisy_z, timesteps=t,
                                context=contexts,
                                context_coords=context_coords,
                                spacings=spacings,
                                down_block_additional_residuals=down_residuals,
                                mid_block_additional_residual=mid_residual)

        # predicted noise with or without conditioning
        noise_pred = predict_noise(seg=self.pre_spade(condition)) if self.spade_cond else predict_noise()

        weights = self.loss_weights(vessels)
        if weights is not None:
            loss = weights * self.loss(noise_pred.float(), noise.float(), reduction='none')
            loss = loss.mean()
        else:
            loss = self.loss(noise_pred.float(), noise.float())

        return loss, z_1, down_residuals, mid_residual

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
        Does the diffusion reverse pass with/without ControlNet conditioning.

        Returns:
            The final latent (and intermediates).
        """
        z_ = z_0
        intermediates = []
        labels = conditions
        if self.spade_cond:
            conditions = self.pre_spade(conditions)

        timesteps = self.scheduler.timesteps.unsqueeze(-1).to(device=z_0.device)
        for t, t_ in zip(self.scheduler.timesteps, timesteps):

            if controlnet:
                down_residuals, mid_residual = controlnet(x=z_,
                                                          timesteps=t_,
                                                          context=contexts,
                                                          context_coords=context_coords,
                                                          controlnet_cond=labels,
                                                          conditioning_scale=cond_scale)
            else:
                down_residuals, mid_residual = None, None

            predict_noise = partial(self.model.forward,
                                    x=z_, timesteps=t_,
                                    context=contexts,
                                    context_coords=context_coords,
                                    spacings=spacings,
                                    down_block_additional_residuals=down_residuals,
                                    mid_block_additional_residual=mid_residual)

            z_t = self.cfg * (predict_noise(seg=conditions) if self.spade_cond else predict_noise())

            # classifier-free-guidance
            # either controlnet or SPADE
            if (1 - self.cfg) != 0.0:
                # simulate empty label map with 0s, same as in DataLoader
                empty = torch.zeros_like(labels).to(labels)

                if controlnet:
                    empty_down_residuals, empty_mid_residual = controlnet(
                        x=z_, timesteps=t_,
                        contronet_cond=empty,
                        context=contexts,
                        context_coords=context_coords,
                        conditioning_scale=cond_scale
                    )
                    z_t += (1.0 - self.cfg) * self.model.forward(
                        x=z_, timesteps=t_,
                        context=contexts,
                        context_coords=context_coords,
                        spacings=spacings,
                        down_block_additional_residuals=empty_down_residuals,
                        mid_block_additional_residual=empty_mid_residual
                    )
                elif self.spade_cond:
                    empty = self.pre_spade(empty)
                    z_t += (1.0 - self.cfg) * predict_noise(seg=empty)

            z_, _ = self.scheduler.step(z_t, t, z_)

            if save_intermediates and (t % self.save_intermediate_steps) == 0:
                intermediates.append(z_)

        if save_intermediates:
            return intermediates
        else:
            return z_
