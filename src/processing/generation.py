from src.generative.base import LatentGenerativeModel
from src.generative.flow.flowmatcher import LatentFlowMatcherModel
from src.generative.diffusion.ddpm import LatentDiffusionModel
from generative.networks.nets.controlnet import ControlNet


class LGMProcessor:
    latent_model: LatentGenerativeModel = None

    @staticmethod
    def init(checkpoint: str, type: str, device):
        """
        Initializes the latent generative model trained during the second stage.

        Args:
            checkpoint: Path to trained model file.
            type: The type of the generative model, one of [flow, diffusion].
            device: device on which to transfer the model
        """
        Model = LatentFlowMatcherModel if type == 'flow' else LatentDiffusionModel
        latent_model = Model.load_from_checkpoint(checkpoint, map_location=device)
        latent_model.freeze()

        LGMProcessor.latent_model = latent_model

    @staticmethod
    def controlnet_init(controlnet: ControlNet, verbose: bool = True):
        """
        Copy the weights from the latent generative model to the ControlNet.
        Adapted from https://github.com/Project-MONAI/GenerativeModels/blob/main/generative/networks/nets/controlnet.py.

        Args:
            controlnet: ControlNet instance.
            verbose: Whether to print missing keys.
        """

        # pytorch lightning adds a prefix during saving, we have to remove it otherwise weights wont be loaded.
        state_dict_frozen = {k.replace("model.", ""): v
                             for k, v in LGMProcessor.latent_model.state_dict().items()}

        output = controlnet.load_state_dict(state_dict_frozen, strict=False)
        if verbose:
            dm_keys = [p[0] for p in list(LGMProcessor.latent_model.named_parameters())
                       if p[0] not in output.unexpected_keys]
            print(
                f"Copied weights from {len(dm_keys)} keys of the diffusion model into the ControlNet:"
                f"\n{'; '.join(dm_keys)}\nControlNet missing keys: {len(output.missing_keys)}:"
                f"\n{'; '.join(output.missing_keys)}\nDiffusion model incompatible keys: {len(output.unexpected_keys)}:"
                f"\n{'; '.join(output.unexpected_keys)}"
            )
