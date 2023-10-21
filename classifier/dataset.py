import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os
import torch
from utils import *


class LungUltrasoundDataset(Dataset):
    def __init__(self, d_type, dataframe, exp_dir, clearml=False, transform=None, debug_mode=False, synthetic_df=None, synthetic_perc=0):
        self.dataframe = dataframe[dataframe['data_type'] == d_type]
        self.transform = transform
        
        # Add synthetic data to the training set
        if d_type == 'train' and synthetic_df is not None:
            self.orig_synthetic_df = synthetic_df
            # tot_viral = len(self.dataframe[self.dataframe['label_name'] == 'viral'])  # Get the number of 'viral' samples
            tot_fake = len(self.orig_synthetic_df)
            if synthetic_perc > 0:
                # n_synth = int((synthetic_perc / 100) * tot_viral)
                n_synth = int((synthetic_perc / 100) * tot_fake)
                self.sampled_synthetic_df = self.orig_synthetic_df.sample(n=n_synth, replace=False)  # Sample with replacement for oversampling
                # Add 'data_type' column to synthetic_df
                self.sampled_synthetic_df['data_type'] = 'train'
                print(f'Added {n_synth} frames to the data.')

                self.sampled_synthetic_df['image_path'] = self.sampled_synthetic_df['image_path'].str.replace('/Users/amitaylev/Desktop/Amitay/Msc/4th semester/ML4Health/Final project/ddpm_experiments/', '/home/lamitay/uls_experiments/ddpm/')

                self.sampled_synthetic_df['fake'] = 1
                self.dataframe['fake'] = 0

                # Select only the relevant columns
                self.dataframe = self.dataframe[['image_path', 'label', 'label_name', 'data_type', 'fake']]
                self.sampled_synthetic_df = self.sampled_synthetic_df[['image_path', 'label', 'label_name', 'data_type', 'fake']]

                # Concatenate the DataFrames
                self.dataframe = pd.concat([self.dataframe, self.sampled_synthetic_df], ignore_index=True)


        print('--------------------------------------------------------------')
        print(f'Created {d_type} dataset with {len(self.dataframe)} frames')
        label_stat = get_column_stats(self.dataframe, 'label_name')
        split_stat = get_column_stats(self.dataframe, 'data_type')
        print("\nClass distribution:\n", label_stat)
        print("\nData split distribution:\n", split_stat)

        # Analyzing distribution by data_type and label_name
        distribution_by_split_and_label = self.dataframe.groupby(['data_type', 'label_name']).size().reset_index(name='count')
        distribution_pivot_table = distribution_by_split_and_label.pivot(index='label_name', columns='data_type', values='count')
        distribution_pivot_table = distribution_pivot_table.fillna(0)
        print("\nData distribution:\n",distribution_pivot_table)

        if debug_mode:
            DEBUG_SIZE = 10
            self.dataframe = self.dataframe.sample(n=DEBUG_SIZE)
            print(f'Debug mode, squeezed {d_type} data from {len(self.dataframe)} to {DEBUG_SIZE}')

        if exp_dir:
            df_out_path = os.path.join(exp_dir, 'dataframes', d_type + f'_synth_perc_{synthetic_perc}_df.csv')
            self.dataframe.to_csv(df_out_path, index=False)
            print(f'Saved {d_type} dataframe to {df_out_path}')

        if clearml:
            report_df_to_clearml(self.dataframe, d_type)
            report_df_to_clearml(distribution_pivot_table, d_type, title='Data distribution')

        print('--------------------------------------------------------------')

        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['image_path']
        label = self.dataframe.iloc[idx]['label']
        image = cv2.imread(img_path)
        meta_data = self.dataframe.iloc[idx]
        if self.transform:
            image = self.transform(image)

        return (image, label), meta_data.to_dict()
    