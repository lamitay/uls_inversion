import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os
import torch
from utils import *


# class PreprocessTransform:
#     def __call__(self, image):
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image = cv2.resize(image, (224, 224))
#         image = np.expand_dims(np.array(image), 0) / 255.0
#         return image
    
# class MinMaxNormalize(object):
#     def __call__(self, tensor):
#         tensor=tensor.float()
#         return (tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor))

class LungUltrasoundDataset(Dataset):
    def __init__(self, d_type, dataframe, exp_dir, clearml=False, transform=None, debug_mode=False):
        self.dataframe = dataframe[dataframe['data_type'] == d_type]
        self.transform = transform

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
            df_out_path = os.path.join(exp_dir, 'dataframes', d_type+'_df.csv')
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
        
        if self.transform:
            image = self.transform(image)

        return image, label
    