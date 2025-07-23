from src.processing.generation import LGMProcessor
from src.generative.base import LatentGenerativeModel


class SemanticControlNet(LatentGenerativeModel):
    """
    ControlNet with semantic conditioning on pre-encoded semantic latents.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.num_classes = self.model.conditioning_embedding_in_channels

        # init controlnet
        LGMProcessor.controlnet_init(self.model)

        # save hyperparameters
        self.save_hyperparameters()

    def _rand_time(self, z_1):
        return LGMProcessor.latent_model._rand_time(z_1)

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
        down_residuals, mid_residual = self.model.forward(x=z_1,
                                                           timesteps=t,
                                                           controlnet_cond=condition)

        # during forward pass with controlnet, the conditioning input is only for the ControlNet
        return LGMProcessor.latent_model._forward(z_1, t, condition=None,
                                                  contexts=contexts,
                                                  context_coords=context_coords, spacings=spacings,
                                                  down_residuals=down_residuals, mid_residual=mid_residual,
                                                  vessels=vessels)

    def _sample_latents(
        self,
        z_0,
        conditions,
        contexts,
        spacings,
        controlnet = None,
        cond_scale: float = 1.0,
        save_intermediates: bool = False
    ):
        return LGMProcessor.latent_model._sample_latents(z_0,
                                                            conditions,
                                                            contexts,
                                                            spacings,
                                                            self.model.forward,
                                                            cond_scale,
                                                            save_intermediates)
