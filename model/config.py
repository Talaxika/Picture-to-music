import json
from dataclasses import dataclass, asdict

@dataclass
class PictureToMusicConfig:
    image_encoder_type: str = 'vit'
    image_encoder_checkpoint: str = 'google/vit-base-patch16-224-in21k'
    audio_encoder_type: str = 'clap'
    audio_encoder_checkpoint: str = 'laion/clap-htsat-unfused'
    freeze_audio_encoder: bool = True
    freeze_image_encoder: bool = True
    num_layers_to_unfreeze: int = 1
    image_embedding_size: int = 768
    audio_embedding_size: int = 512
    shared_embedding_size: int = 512
    has_audio_mapper: bool = False
    mlp_hidden_size: int = 1024

def serialize_config(filename):
    config = PictureToMusicConfig()
    with open(filename, 'w') as f:
        json.dump(asdict(config), f)
