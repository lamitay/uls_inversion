import os
import pandas as pd
from PIL import Image
import argparse
import random
import numpy as np

def extract_info(filename):
    """
    Extract sequence name and frame number from a given filename.

    :param filename: Name of the file
    :return: Tuple containing sequence name and frame number
    """
    # Split the filename based on '_frame' or '_image' to extract sequence name and frame number
    frame_num_part = ""
    if '_frame' in filename:
        seq_name, _, frame_num_part = filename.rpartition('_frame')
    elif '_image' in filename:
        seq_name, _, frame_num_part = filename.rpartition('_image')
    else:
        seq_name = filename.split('.')[0]

    frame_num = 0
    if frame_num_part:
        frame_num_part = frame_num_part.split('.')[0]
        if frame_num_part.isdigit():
            frame_num = int(frame_num_part)
    else:
        # Extract all digits at the end of the sequence name
        seq_name_base, frame_num_part = seq_name.rstrip('0123456789'), seq_name[len(seq_name.rstrip('0123456789')):]
        if frame_num_part.isdigit():
            frame_num = int(frame_num_part)
            seq_name = seq_name_base

    return seq_name, frame_num

def create_splits_with_handling(df, seq_column, label_column, train_size, val_size):
    """
    Create train, validation, and test splits, handling small classes.

    :param df: DataFrame containing the data
    :param seq_column: Column name for sequence identifier
    :param label_column: Column name for labels
    :param train_size: Proportion of training data
    :param val_size: Proportion of validation data
    :return: Lists containing training, validation, and test sequences
    """
    # Initialize lists for sequences
    train_seq = []
    val_seq = []
    test_seq = []
    # Iterate through groups by label
    for label, group in df.groupby(label_column):
        unique_sequences = group[seq_column].unique()
        total_sequences = len(unique_sequences)
        # Handle small classes differently
        if total_sequences <= 3:
            shuffled_sequences = list(np.random.permutation(unique_sequences))
            train_seq.append(shuffled_sequences.pop(0))
            if shuffled_sequences:
                val_seq.append(shuffled_sequences.pop(0))
            test_seq.extend(shuffled_sequences)
        else:
            # Shuffle and split sequences for train, validation, and test
            num_train = int(total_sequences * train_size)
            num_val = int(total_sequences * val_size)
            shuffled_sequences = np.random.permutation(unique_sequences)
            train_seq.extend(shuffled_sequences[:num_train])
            val_seq.extend(shuffled_sequences[num_train:num_train + num_val])
            test_seq.extend(shuffled_sequences[num_train + num_val:])
    return train_seq, val_seq, test_seq

def create_data_df(images_dir):
    """
    Create a DataFrame containing image data, including file path, sequence name, frame number, and label.

    :param images_dir: Directory containing the images
    :return: DataFrame with image data
    """
    # Initialize data and label dictionary
    data = []
    labels_dict = {}
    label_num = 0
    # Iterate through labels and images
    for label_name in os.listdir(images_dir):
        label_path = os.path.join(images_dir, label_name)
        if os.path.isdir(label_path):
            if label_name not in labels_dict:
                labels_dict[label_name] = label_num
                label_num += 1
            for image_name in os.listdir(label_path):
                image_path = os.path.join(label_path, image_name)
                seq_name, frame_num = extract_info(image_name)
                width, height = Image.open(image_path).size
                data.append([image_name, image_path, seq_name, frame_num, label_name, labels_dict[label_name], width, height])

    # Create DataFrame and add data_type based on custom splits
    df = pd.DataFrame(data, columns=['image_name', 'image_path', 'seq_name', 'frame_num', 'label_name', 'label', 'width', 'height'])
    train_seq, val_seq, test_seq = create_splits_with_handling(df, 'seq_name', 'label_name', 0.7, 0.15)  # Custom splits
    df['data_type'] = 'test'
    df.loc[df['seq_name'].isin(train_seq), 'data_type'] = 'train'
    df.loc[df['seq_name'].isin(val_seq), 'data_type'] = 'val'

    return df

if __name__ == "__main__":
    # Argument parsing for command-line execution
    parser = argparse.ArgumentParser(description="Create a DataFrame from lung ultrasound data.")
    parser.add_argument("--images_dir", default="/home/lamitay/vscode_projects/covid19_ultrasound/data/image_dataset", help="Path to the directory containing the images.")
    args = parser.parse_args()
    df = create_data_df(args.images_dir)

    # Analyzing distribution by data_type and label_name
    distribution_by_split_and_label = df.groupby(['data_type', 'label_name']).size().reset_index(name='count')
    distribution_pivot_table = distribution_by_split_and_label.pivot(index='label_name', columns='data_type', values='count')
    distribution_pivot_table = distribution_pivot_table.fillna(0)
    print(distribution_pivot_table)

    # Saving distribution and DataFrame to CSV
    distribution_csv_file_name = 'lung_uls_distribution.csv'
    distribution_csv_file_path = os.path.join(args.images_dir,distribution_csv_file_name)
    distribution_pivot_table.to_csv(distribution_csv_file_path)
    print(f"Distribution saved to {distribution_csv_file_path}")

    output_path = os.path.join(args.images_dir, 'lung_uls_data.csv')
    df.to_csv(output_path, index=False)
    print(f"DataFrame saved to {output_path}")
