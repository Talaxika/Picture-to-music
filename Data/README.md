# Dataset Preparation

The IMEMNet dataset was used for the data preparation. Following the README from https://github.com/linkAmy/IMEMNet/tree/master we downloaded both the Music DEAM dataset and the EMOTIC image dataset. NAPS and IAPS are not used, as they require access. The EMOTIC dataset itself consists of more than 70% of the whole IMEMNet dataset.

The matching files had the real iamge names. The EMOTIC dataset was downloaded from here: https://www.kaggle.com/datasets/magdawjcicka/emotic. It is already preprocessed, and can be directly used using kaggle. The annots_arr files explain how the images were preprocessed and how they were renamed.

That's why we had to modify the matching files with the new names, as well as remove the missing IAPS and NAPS images from the dataset. The process involved normalizing image IDs to a consistent format, mapping original image filenames to their corresponding .npy feature files, and filtering the matching pairs to only include those with available annotations. This resulted in a cleaned matching file linking audio IDs to image feature paths along with their matching scores, ready for training and evaluation.

For the purpose of the task, the pairs with less than 0.5 matching score were also removed, as it is not classification, but matching/generation, where we need quality data.

# Quick overview of process

## Datasets
Images (from the EMOTIC dataset via Kaggle)
Audio (from the DEAM dataset, local .mp3s)
Matching Scores (precomputed text files linking image-audio pairs)

## Inside datasets
Image features preprocessed and stored as .npy tensors (224×224×3)
Raw audio .mp3 files for each audio ID
Matching files (cleaned_train/val/test_matching.txt) listing:
    <audio_id> <image_file.npy> <score>

## How to works
1. Mount Google Drive and unzip the project_data.zip containing audio and matching files.
2. Download EMOTIC dataset via kagglehub.
3. Load matching files into DataFrames.
4. Use a custom PyTorch Dataset (ImageAudioDataset) that:
    - Loads and normalizes image tensors
    - Loads and resamples audio to a fixed length (e.g. 50s)
    - Pads or truncates waveforms to fixed length for batching
    - Returns image + audio + score + ID

## How to use
Create dataset and loader (default 50s, 44.1kHz)
train_dataset = ImageAudioDataset(df_train, img_dir, audio_dir)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

## Customize:
train_dataset = ImageAudioDataset(df_train, img_dir, audio_dir, sr=22050, max_sec=35)

## Each batch
{
  "image":       torch.Tensor [B, 3, 224, 224],
  "audio_raw":   torch.Tensor [B, sr * max_sec],
  "score":       torch.Tensor [B],
  "audio_id":    list of strings
}

## Setup
1. Download and unzip Deam audio: https://cvml.unige.ch/databases/DEAM/
2. copy the contents of DEAM_audio\MEMD_audio inside music/
3. Compress
4. Place project_data.zip in your Google Drive → contains:
    - /music/*.mp3
    - cleaned_train_matching.txt
    - cleaned_test_matching.txt
    - cleaned_val_matching.txt

5. Upload kaggle.json to authenticate kagglehub when asked in notebook

## Applications
clean_data.py was used to clean up the matching files, by removing missing images and filtering by scores.

data_preparation.ipynb is the notebook with the above mentioned work inside.