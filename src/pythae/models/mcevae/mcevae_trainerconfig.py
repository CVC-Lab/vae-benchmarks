from pydantic.dataclasses import dataclass

@dataclass
class MCEVAE_trainerConfig():
	num_epochs: int = 50,
	learning_rate: float =1e-3,
	batch_size: int =200
