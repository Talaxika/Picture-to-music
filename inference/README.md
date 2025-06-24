# Picture-to-Music Inference API
This repo has code and a pretrained model that connects images to music, like CLIP but for pictures and songs. It uses an image input, and it finds music from the FMA dataset that matches the vibe or emotion.

## How does it work
Image encoder: Uses a Vision Transformer (ViT) to turn images into embeddings.

Audio encoder: Uses a CLAP model to embed audio.

Shared space: Both are mapped to the same space using MLPs.

Retrieval: When you upload an image, it finds the closest music tracks from the precomputed FMA embeddings.

## How to use
1. Open the notebook on Kaggle.
2. Attach the FMA dataset from Kaggle’s “Add Data” sidebar (if not already present).
    - Dataset: FMA Free Music Archive (small + medium)
3. Enable GPU.
4. Run all cells.
    - The default audio_dir is set to Kaggle’s dataset path, so no extra setup is needed.

## Pretrained Model

- Model and config are automatically downloaded from Hugging Face:
https://huggingface.co/Pesho564/Picture-to-music

## Example Output
When you run the retrieval cell, you get the filenames/paths of the top matching music tracks for your input image.