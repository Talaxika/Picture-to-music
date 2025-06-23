import pandas as pd
import os

matching_file_path_train = './data/IMEMNet_matchings/train_matching.txt'
output_path_train = './data/IMEMNet_matchings/cleaned_train_matching.txt'

matching_file_path_val = './data/IMEMNet_matchings/val_matching.txt'
output_path_val = './data/IMEMNet_matchings/cleaned_val_matching.txt'

matching_file_path_test = './data/IMEMNet_matchings/test_matching.txt'
output_path_test = './data/IMEMNet_matchings/cleaned_test_matching.txt'

def get_clean_matchings(matching_file_path, output_path):
    EMOTIC_ANN_MANIFEST_PATHS = [
        './data/Images/archive/annots_arrs/annot_arrs_val.csv',
        './data/Images/archive/annots_arrs/annot_arrs_train.csv',
        './data/Images/archive/annots_arrs/annot_arrs_extra_train.csv',
        './data/Images/archive/annots_arrs/annot_arrs_test.csv',
    ]
    NPY_FILES_ROOT = './data/Images/archive/img_arrs'

    image_id_to_npy_path_map = {}

    for manifest_path in EMOTIC_ANN_MANIFEST_PATHS:
        try:
            emotic_manifest_df = pd.read_csv(manifest_path)
            print(f"Loaded: {os.path.basename(manifest_path)} ({len(emotic_manifest_df)} rows)")

            if 'Filename' not in emotic_manifest_df.columns:
                print(f"'Filename' column not found in {manifest_path}")
                continue

            if 'Crop_name' not in emotic_manifest_df.columns and 'Arr_name' not in emotic_manifest_df.columns:
                print(f"No 'Crop_name' or 'Arr_name' in {manifest_path}")
                continue

            npy_col = 'Crop_name' if 'Crop_name' in emotic_manifest_df.columns else 'Arr_name'

            for _, row in emotic_manifest_df.iterrows():
                filename = str(row['Filename'])
                npy_filename = str(row[npy_col])
                full_path = os.path.join(NPY_FILES_ROOT, npy_filename)
                if os.path.exists(full_path):
                    image_id_to_npy_path_map[filename] = npy_filename
                else:
                    continue

        except Exception as e:
            print(f"Error processing {manifest_path}: {e}")

    print("\nLoading and normalizing matching file...")
    try:
        df_matchings = pd.read_csv(matching_file_path, sep=' ', header=None, names=['audio_id', 'original_image_id', 'matching_score'])
        print(f"Loaded: {matching_file_path} ({len(df_matchings)} rows)")
    except Exception as e:
        print(f"Failed to load matching file: {e}")
        exit()

    def normalize_image_id(image_id):
        s = str(image_id)
        return s if s.endswith('.jpg') else s + '.jpg'

    df_matchings['original_image_id'] = df_matchings['original_image_id'].apply(normalize_image_id)

    print("\nMapping to NPY paths...")
    df_matchings['npy_image_path'] = df_matchings['original_image_id'].map(image_id_to_npy_path_map)

    df_final = df_matchings.dropna(subset=['npy_image_path']).copy()
    print(f"→ Matches after filtering: {len(df_final)}")

    df_final = df_final[df_final['matching_score'] >= 0.5].copy()
    print(f"→ Matches after filtering by matching_score >= 0.5: {len(df_final)}")

    final_df = df_final[['audio_id', 'npy_image_path', 'matching_score']]
    final_df.rename(columns={'npy_image_path': 'image_path'}, inplace=True)

    print("\nSample (after mapping):")
    print(final_df.head().to_string())

    final_df.to_csv(output_path, sep=' ', index=False, header=False)
    print(f"\nSaved cleaned matching file to: {output_path}")

# Note: To generate clean matching for each split, run this with the corresponding arguments
get_clean_matchings(matching_file_path_train, output_path_train)
#get_clean_matchings(matching_file_path_val, output_path_val)
#get_clean_matchings(matching_file_path_test, output_path_test)