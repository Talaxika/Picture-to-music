# Picture-to-music

## Overview
Picture-to-music is a CLIP-inspired model that encodes semantically in a shared embedding space both images and music.\
There is an image encoder just like in CLIP and an audio encoder.We use the cross entropy contrastive loss from the [CLIP paper](https://arxiv.org/abs/2103.00020).
The whole goal of the model is to learn how to represent images and audio in a shared space.\
We hope, that at least the model captures the emotional correspondence between the samples.\
For the used dataset, check out the [data folder](https://github.com/Talaxika/Picture-to-music/tree/main/data).

## Code and weights

The [github repo](https://github.com/Talaxika/Picture-to-music/tree/main) consists of the training and inference code for the model.\
...\
The [hugging face repo](https://huggingface.co/Pesho564/Picture-to-music) has the:
  - *model weights* - `model_state_dict.bin`
  - *model config file* (it is used to initialize the model in code with the right hyperparams) - `config.json`

To use the model for inference, download the `model_state_dict.bin` and `config.json` files.
Then you can inference the model like this:
```
TODO
```