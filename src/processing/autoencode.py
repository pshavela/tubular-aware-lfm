import warnings

from src.autoencoder.vae import VQVAEModel
from monai.data.meta_tensor import MetaTensor


class VAEProcessor:
    vqvae: VQVAEModel = None
    normalize: bool = False
    concat_decode: bool = False

    @staticmethod
    def init(checkpoint: str, device, load: str = 'full', normalize: bool = False):
        """
        Initializes the VQVAE instance.

        Args:
            checkpoint: Path to VQVAE checkpoint file.
            device: device on which to transfer VQVAE.
            load: One of ['full', 'decoder', 'encoder']
        """
        assert load in ['full', 'decoder', 'encoder']

        # hide warning that loss component of VQVAE is not loaded
        warnings.filterwarnings('ignore', '.*Found keys that are not in the model state dict but in the checkpoint*')
        # load VAE model without loss component and freeze
        vqvaemodel = VQVAEModel.load_from_checkpoint(checkpoint, map_location=device, loss_config={}, strict=False)
        vqvaemodel.freeze()

        if load == 'decoder':
            del vqvaemodel.vqvae.encoder
        elif load == 'encoder':
            del vqvaemodel.vqvae.decoder

        VAEProcessor.vqvae = vqvaemodel
        VAEProcessor.normalize = normalize
        VAEProcessor.concat_decode = vqvaemodel.concat_vessels

    @staticmethod
    def encode(images: MetaTensor) -> MetaTensor:
        """
        Encodes images to the latent representations with the VQVAE model. Used as a preprocessing step.
        The latents will be in raw form, ie. not quantized. quantization is applied when decoding.

        Args:
            images: input of shape BCHW[D], where B is batch size, C is number of channels and HW[D]
                are spatial dimensions.

        Returns:
            raw latents of shape BLH'W'[D'], where L is the codebook vector dimension.
        """
        # raw latents batch, not codebook vectors -> decoder absorbs quantization layer
        z = VAEProcessor.vqvae.encode_stage_2_inputs(images.to(VAEProcessor.vqvae.device), normalize=VAEProcessor.normalize)
        return z.to(images.device)

    @staticmethod
    def decode(z: MetaTensor, concat: MetaTensor = None) -> MetaTensor:
        """
        Decodes the latent images with the VQVAE model. Used as a postprocessing step.
        The latents will be quantized before decoding.

        Args:
            z: latents of shape BLH'W'[D'].

        Returns:
            images of shape BCHW[D].
        """
        images = VAEProcessor.vqvae.decode_stage_2_outputs(z.to(VAEProcessor.vqvae.device),
                                                           denormalize=VAEProcessor.normalize,
                                                           concat=concat)
        return images.to(z.device)

    @staticmethod
    def autoencode(images: MetaTensor, concat: MetaTensor = None) -> MetaTensor:
        """
        Autoencodes input.

        Args:
            images: Input images.

        Returns:
            Reconstructed images.
        """
        latents = VAEProcessor.encode(images.to(VAEProcessor.vqvae.device))
        return VAEProcessor.decode(latents, concat).to(images.device)

    @staticmethod
    def cleanup(keep: str = None):
        """
        Removes model from GPU.

        Args:
            keep: one of [None, 'encoder', 'decoder']. Whether to keep a certain part of the VQVAE.
        """
        assert keep in [None, 'decoder', 'encoder']

        if VAEProcessor.vqvae is None:
            return

        if not keep:
            del VAEProcessor.vqvae
        elif keep == 'decoder':
            del VAEProcessor.vqvae.vqvae.encoder
        else:
            del VAEProcessor.vqvae.vqvae.decoder
