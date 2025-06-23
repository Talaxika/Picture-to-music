from huggingface_hub import hf_hub_download
from picture_to_music import PictureToMusicModel, PictureToMusicConfig
import torch, json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load config from hugging face
config_path = hf_hub_download("Pesho564/Picture-to-music", "config.json")
with open(config_path) as f:
     config = json.load(f)

config_class = PictureToMusicConfig(**config)

# Load weights
weights_path = hf_hub_download("Pesho564/Picture-to-music", "model_state_dict.bin")
model = PictureToMusicModel(config_class).to(device)
model.load_state_dict(torch.load(weights_path))

model.eval()

# Model is now ready