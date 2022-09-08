from pydantic.dataclasses import dataclass

from ..vae import VAEConfig


@dataclass
class MCEVAEConfig(VAEConfig):
    r"""
    Disentangled :math:`\MCE`-VAE model config config class

    Parameters:
        input_dim (tuple): The input_data dimension.
        latent_dim (int): The latent space dimension. Default: None.
        reconstruction_loss (str): The reconstruction loss to use ['bce' 'mse']. Default: 'mse'
        beta (float): The balancing factor. Default: 10.
        C (float): The value of the KL divergence term of the ELBO we wish to approach, measured in
            nats. Default: 50.
        warmup_epoch (int): The number of epochs during which the KL divergence objective will
            increase from 0 to C (should be smaller or equal to nb_epochs). Default: 100
        epoch (int): The current epoch. Default: 0
    """
    in_size: int = 28*28
    aug_dim: float = 16*7*7
    latent_z_c: int = 0
    latent_z_var: int = 5
    mode: str = 'SO2'
    invariance_decoder: str ='gated'
    rec_loss: str ='mse'
    div: str='KL'
    in_dim: int = 1
    out_dim: int = 1
    hidden_z_c: int = 300
    hidden_z_var: int =300
    hidden_tau: int =32 
    training_mode: str = 'supervised'
    device: str = 'cpu'
    tag: str = 'default'
