# Dataset Preparation

The IMEMNet dataset was used for the data preparation.\
Following the README from the [IMEMNET repo](https://github.com/linkAmy/IMEMNet/tree/master), we downloaded both the Music DEAM dataset and the EMOTIC image dataset. NAPS and IAPS are not used, as they require access. The EMOTIC dataset itself consists of more than 70% of the whole IMEMNet dataset.

The matching files had the real image names. The EMOTIC dataset was downloaded from [kaggle](https://www.kaggle.com/datasets/magdawjcicka/emotic). It is already preprocessed, and can be directly used. The annots_arr files explain how the images were preprocessed and how they were renamed.

That's why we had to modify the matching files with the new names, as well as remove the missing IAPS and NAPS images from the dataset. The process involved normalizing image IDs to a consistent format, mapping original image filenames to their corresponding .npy feature files, and filtering the matching pairs to only include those with available annotations. This resulted in a cleaned matching file linking audio IDs to image feature paths along with their matching scores, ready for training and evaluation.

For the purpose of the task, the pairs with less than a specified matching score were also removed, as it is not classification, but matching/generation, where we need quality data.

# Setup
## Setup in Google Colab
1. Download the [project_data](https://github.com/Talaxika/Picture-to-music/tree/main/Data/project_data) folder locally
2. Download the [DEAM dataset](https://cvml.unige.ch/databases/DEAM/) (only the audio)
3. Extract the contents of DEAM_audio/MEMD_audio inside project_data/music/
3. Zip the resulting project_data directory
4. Upload project_data.zip to your Google Drive. It must contain:
    - /music/*.mp3
    - cleaned_train_matching.txt
    - cleaned_test_matching.txt
    - cleaned_val_matching.txt
5. Upload kaggle.json to Colab to authenticate with kagglehub when asked in the notebook

## Setup in Kaggle
1. After opening the notebook, add the emotic dataset as an input (directly from the kaggle UI)
2. No additional steps :smile:

# Quick overview of the data cleaning process

## Datasets
  - Images - the EMOTIC dataset via Kaggle
  - Audio - the DEAM dataset, local .mp3s
  - Matching Scores - precomputed text files linking image-audio pairs

## Inside the datasets
  - Image features are preprocessed and stored as .npy tensors (224×224×3)
  - Raw audio .mp3 files for each audio ID
  - Matching files (cleaned_train/val/test_matching.txt) listing: `<audio_id> <image_file.npy> <score>`

## How the data is loaded (Colab)
1. Google Drive is mounted and `project_data.zip` is unzipped
2. The EMOTIC dataset is downloaded via kagglehub.
3. The matching files are loaded into DataFrames.
4. We create a custom PyTorch Dataset class (ImageAudioDataset) that:
    - Loads image tensors
    - Loads and resamples audio to a fixed length (e.g. 50s)
    - Pads or truncates waveforms to fixed length for batching (randomly crops also)
    - Returns image + audio + score + ID

## How to use and customize dataset
Creating a dataset and loader (default 50s, 44.1kHz)
```python
train_dataset = ImageAudioDataset(df_train, img_dir, audio_dir, sr=22050, max_sec=35)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

## Each batch looks like this
```python
{
  "image":       # torch.Tensor [B, 3, 224, 224],
  "audio_raw":   # torch.Tensor [B, sampling_rate * max_sec],
  "score":       # torch.Tensor [B],
  "audio_id":    # list of strings
}
```

## Applications
clean_data.py was used to clean up the matching files, by removing missing images and filtering by scores.

data_preparation.ipynb is the notebook with the above mentioned work inside.