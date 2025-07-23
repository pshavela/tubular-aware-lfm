import torch
from torch import nn

from generative.networks.nets.patchgan_discriminator import PatchDiscriminator
from generative.losses.perceptual import PerceptualLoss
from generative.losses.adversarial_loss import PatchAdversarialLoss
from generative.losses.spectral_loss import JukeboxLoss as SpectralLoss

class VQGANLoss(nn.Module):
    """
    The loss for the VQVAE module based on the VQGAN paper https://arxiv.org/abs/2012.09841
    and Morphology-preserving Autoregressive 3D Generative Modelling of the Brain by Tudosiu et al.
    (https://arxiv.org/pdf/2209.03177.pdf).

    Args:
        pixel_loss: either "l1" or "l2"
        adverserial_weight: the weight factor used for the adverserial loss
        perceptual_weight: the weight factor used for the LPIPS loss
        spectral_weight: the weight factor used for the Spectral Loss
        discriminator_weight: the weight factor used for the adaptive weighting scheme. Only in conjuction with
            adaptive_lambda is True
        discriminator_step_start: at which step to involve the adverserial loss with the discriminator
        wasserstein_loss: whether to set the adverserial loss to the wasserstein objective. Based on the paper
            https://arxiv.org/abs/1704.00028. Will use the adaptive weighting scheme from the VQGAN paper.
        discriminator_config: configuration for the patch discriminator network
        perceptual_config: configuration for the LPIPS network
    """
    def __init__(self,
                 *,
                 spatial_dims: int,
                 pixel_loss: str = 'l1',
                 adverserial_weight: float = 1.0,
                 perceptual_weight: float = 1.0,
                 spectral_weight: float = 1.0,
                 vessel_weight: float = 0.0,
                 quantization_weight: float = 1.0,
                 disriminator_step_start: int = 10000,
                 wasserstein_loss: bool = False,
                 discriminator_config,
                 perceptual_config):
        super().__init__()
        if pixel_loss == 'l1':
            self.pixel_loss = nn.L1Loss()
        elif pixel_loss == 'l2':
            self.pixel_loss = nn.MSELoss()
        else:
            raise ValueError('Unknown per-pixel loss function. Must be one of ["l1", "l2"]')

        self.perceptual_weight = perceptual_weight
        self.adverserial_weight = adverserial_weight
        self.spectral_weight = spectral_weight
        self.vessel_weight = vessel_weight
        self.quantization_weight = quantization_weight
        self.discriminator_step_start = disriminator_step_start
        self.wasserstein_loss = wasserstein_loss

        self.perceptual_loss = PerceptualLoss(spatial_dims=spatial_dims, **perceptual_config)
        self.spectral_loss = SpectralLoss(spatial_dims=spatial_dims)
        self.discriminator = PatchDiscriminator(spatial_dims=spatial_dims, **discriminator_config)
        self.adverserial_loss = PatchAdversarialLoss(criterion='least_squares')

        # somehow PerceptualLoss with pretrained LPIPS does not set requires_grad to False
        for param in self.perceptual_loss.parameters():
            param.requires_grad = False

    def adaptive_weight(self, loss_reconstruct, loss_gan, decoder_last_layer_weight):
        """
        Adaptive weight computation as per https://github.com/CompVis/taming-transformers

        Args:
            loss_reconstruct: The reconstruction loss.
            loss_gan: generator part of the adverserial loss.
            decoder_last_layer_weight: The weights of the last layer of the VQVAE decoder.

        Returns:
            Lambda, the weight factor for loss_gan.
        """

        if not self.wasserstein_loss:
            return torch.tensor(1.0)

        rloss_grads = torch.autograd.grad(loss_reconstruct, decoder_last_layer_weight, retain_graph=True)[0]
        gloss_grads = torch.autograd.grad(loss_gan, decoder_last_layer_weight, retain_graph=True)[0]

        lamb = torch.norm(rloss_grads) / (torch.norm(gloss_grads) + 1e-4)
        lamb = torch.clamp(lamb, 0, 1e4).detach()
        return lamb

    def forward(self,
                input: torch.Tensor,
                reconstruction: torch.Tensor,
                quantization_loss: torch.Tensor,
                decoder_last_layer_weight: torch.Tensor,
                vessel_mask: torch.Tensor,
                global_step,
                optimize_over: str,
                split: str):
        """
        Computes the loss for each the generator (VQVAE) and the Patch discriminator. Based on the VQ-GAN loss
        from https://github.com/CompVis/taming-transformers and the paper https://arxiv.org/pdf/2112.10752, equation 25

        Args:
            input: batch of input images.
            reconstruction: batch of reconstruced images.
            quantization loss: Loss of the quantization to the codebook vectors.
            decoder_last_layer_weight: The weights of the last layer of the VQVAE decoder. Used for the Wasserstein loss.
            global_step: Current train iteration.
            optimize_over: One of ["generator", "discriminator"]. Determines which loss should be computed.
            split: One of ["train", "val"]. Used for logging.

        Returns:
            Either the generator or the discriminator loss.
        """

        if optimize_over == 'generator':
            loss_reconstruct = self.pixel_loss(reconstruction.float(), input.float())

            loss_vessels = torch.tensor(0.0)
            if vessel_mask is not None and self.vessel_weight > 0:
                loss_vessels = self.vessel_weight * self.pixel_loss(
                    (vessel_mask * reconstruction).float(),
                    (vessel_mask * input).float()
                )
                loss_reconstruct += loss_vessels

            loss_perceptual = self.perceptual_loss(reconstruction.float(), input.float())
            loss_reconstruct += self.perceptual_weight * loss_perceptual

            # Patch discriminator
            logits_fake = self.discriminator(reconstruction.contiguous().float())[-1]

            if self.wasserstein_loss:
                loss_gan = -torch.mean(logits_fake)
            else:
                loss_gan = self.adverserial_loss(logits_fake, target_is_real=True, for_discriminator=False)

            loss_generator = loss_reconstruct + (self.quantization_weight * quantization_loss)
            # adverserial loss kicks in after the predefined iteration count
            lamb = torch.tensor(0.0)
            if (global_step >= self.discriminator_step_start) and (split == 'train'):
                lamb = self.adaptive_weight(loss_reconstruct, loss_gan, decoder_last_layer_weight)
                loss_generator += lamb * self.adverserial_weight * loss_gan

            loss_spectral = (self.spectral_weight * self.spectral_loss(input.float(), reconstruction.float())
                             if self.spectral_weight > 0 else torch.tensor(0.0))

            loss_generator += loss_spectral

            return loss_generator, {
                f'{split}/loss_reconstruction': loss_reconstruct,
                f'{split}/loss_vessels': loss_vessels,
                f'{split}/loss_perceptual': loss_perceptual,
                f'{split}/loss_spectral': loss_spectral,
                f'{split}/loss_quantization': quantization_loss,
                f'{split}/loss_generator': loss_generator,
                f'{split}/loss_adverserial': loss_gan,
                f'{split}/lambda': lamb,
            }
        else:
            # optimize over discriminator
            logits_fake = self.discriminator(reconstruction.contiguous().detach())[-1]
            loss_d_fake = self.adverserial_loss(logits_fake, target_is_real=False, for_discriminator=True)
            logits_real = self.discriminator(input.contiguous().detach())[-1]
            loss_d_real = self.adverserial_loss(logits_real, target_is_real=True, for_discriminator=True)

            if self.wasserstein_loss:
                loss_discriminator = torch.mean(torch.nn.functional.softplus(-logits_real))
                loss_discriminator += torch.mean(torch.nn.functional.softplus(logits_fake))
            else:
                loss_discriminator = loss_d_fake + loss_d_real

            loss_discriminator *= 0.5

            # adverserial loss kicks in after the predefined iteration count
            loss_discriminator *= (global_step >= self.discriminator_step_start) * self.adverserial_weight

            return loss_discriminator, {
                f'{split}/loss_discriminator': loss_discriminator
            }
