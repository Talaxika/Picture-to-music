# Picture-to-music

## Overview
Picture-to-music is a CLIP-inspired model that encodes semantically in a shared embedding space both images and music.\
There is an image encoder just like in CLIP and an audio encoder.We use the cross entropy contrastive loss from the [CLIP paper](https://arxiv.org/abs/2103.00020).
The whole goal of the model is to learn how to represent images and audio in a shared space.\
We hope, that at least the model captures the emotional correspondence between the samples.\
For the used dataset, check out the [data folder](https://github.com/Talaxika/Picture-to-music/tree/main/data).

## Code and weights

The [github repo](https://github.com/Talaxika/Picture-to-music/tree/main) consists of the training and inference code for the model:
  - *training_kaggle.ipynb* - the training code + data loading but only for kaggle (it's basically the same for colab, follow the guide in the [data folder](https://github.com/Talaxika/Picture-to-music/tree/main/data))
  - *inference/* - this folder contains an example notebook and more information on inferencing the model and using it for a music retrieval task


The [hugging face repo](https://huggingface.co/Pesho564/Picture-to-music) has the:
  - *model weights* - `model_state_dict.bin`
  - *model config file* (it is used to initialize the model in code with the right hyperparams) - `config.json`

To use the model for inference, download the `model_state_dict.bin` and `config.json` files.
Then you can inference the model like this:
```python
from huggingface_hub import hf_hub_download
from picture_to_music import PictureToMusicModel, PictureToMusicConfig
import torch, json

# Load config from hugging face
config_path = hf_hub_download("Pesho564/Picture-to-music", "config.json")
with open(config_path) as f:
     config = json.load(f)

config_class = PictureToMusicConfig(**config)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load weights
weights_path = hf_hub_download("Pesho564/Picture-to-music", "model_state_dict.bin")
model = PictureToMusicModel(config_class).to(device)
model.load_state_dict(torch.load(weights_path))

model.eval()

# Model is now ready
```