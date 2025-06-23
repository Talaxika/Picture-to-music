import torch
from picture_to_music import PictureToMusicModel

# Load config
import json
with open("config.json") as f:
    config = json.load(f)

model = PictureToMusicModel(config)

# Load weights
model.load_state_dict(torch.load("model_state_dict.bin"))

print(model)

# from huggingface_hub import hf_hub_download
# import torch, json

# # Load config for v2.0
# config_path = hf_hub_download("your-username/your-model", "config.json", revision="v2.0")
# with open(config_path) as f:
#     config = json.load(f)

# # Load weights
# weights_path = hf_hub_download("your-username/your-model", "pytorch_model.bin", revision="v2.0")
# model = MyModel(**config)
# model.load_state_dict(torch.load(weights_path))
# model.eval()