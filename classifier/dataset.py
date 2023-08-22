import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os
from utils import *


class PreprocessTransform:
    def __call__(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = np.expand_dims(np.array(image), 0) / 255.0
        return image

class LungUltrasoundDataset(Dataset):
    def __init__(self, d_type, dataframe, exp_dir, clearml_task=False, transform=None, debug_mode=False):
        self.dataframe = dataframe
        self.transform = transform

        print('--------------------------------------------------------------')
        print(f'Created {d_type} dataset with {len(self.dataframe)} frames')

        if debug_mode:
            DEBUG_SIZE = 10
            self.dataframe = self.dataframe.sample(n=DEBUG_SIZE)
            print(f'Debug mode, squeezed {d_type} data from {len(self.dataframe)} to {DEBUG_SIZE}')

        if exp_dir:
            df_out_path = os.path.join(exp_dir, 'dataframes', d_type+'_df.csv')
            self.dataframe.to_csv(df_out_path, index=False)
            print(f'Saved {d_type} dataframe to {df_out_path}')

        if clearml_task:
            report_df_to_clearml(self.dataframe, clearml_task, d_type)

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


class AF_dataset(Dataset):
    """Dataset for binary classification of AF
    """
    def __init__(self, 
                 dataset_folder_path, 
                 record_names, 
                 clearml_task = False,
                 exp_dir = None, 
                 transform = False, 
                 config=None, 
                 d_type='No data type specified', 
                 GAN_label=-1,
                 negative_class_prec = None):
        super().__init__()

        self.transform = transform
        self.folder_path = dataset_folder_path
        # Load meta data csv file from dataset folder:
        meta_data = pd.read_csv(os.path.join(dataset_folder_path,'meta_data.csv')) 
        meta_data = drop_unnamed_columns(meta_data)
        if not record_names[0].endswith('.dat'):
            record_names = [name + '.dat' for name in record_names]
        self.meta_data = meta_data[meta_data['record_file_name'].isin(record_names)]
        print('--------------------------------------------------------------')
        print(f'created {d_type} dataset with {len(self.meta_data)} intervals')

        if GAN_label >= 0:
            self.meta_data = self.meta_data[self.meta_data['label'] == GAN_label]
            print(f'GAN {d_type} dataloader with labels {GAN_label} size is: {len(self.meta_data)}')

        if config is not None:
            # if data quality threshold is provided, remove signals that has a bsqi below threshold
            if isinstance(config['bsqi_th'], float):
                data_quality_thr = config['bsqi_th']
                pre_bsqi_size = len(self.meta_data)
                filter_series = pd.Series(self.meta_data['bsqi_score'] > data_quality_thr)
                self.meta_data = self.meta_data[filter_series] 
                print(f'Using bsqi scores to filter {d_type} dataset with threshold of {data_quality_thr}')
                print(f"bsqi filtered {d_type} from {pre_bsqi_size} to {len(self.meta_data)}") 
                assert len(self.meta_data) > 0 ,'The bsqi filtering filtered all the samples, please choose a lower threshold and run again'

            if negative_class_prec:
                num_negative_class_intervals = sum(self.meta_data['label']==0.)
                num_positive_class_intervals = sum(self.meta_data['label']==1.)
                
                num_wanted_positive_class_intervals = (1/negative_class_prec)*num_negative_class_intervals - num_negative_class_intervals
                if num_positive_class_intervals > num_wanted_positive_class_intervals:
                     # if the number of positive samples is smaller the the wanted number of positive samples, do not filter!
                    pos_indices = np.where(self.meta_data['label']==1.)[0]
                    pos_indices_to_remove = random.sample(pos_indices.tolist(), int(num_positive_class_intervals - num_wanted_positive_class_intervals))
                    pos_indices_to_remove = [self.meta_data.index[i] for i in pos_indices_to_remove]
                    self.meta_data = self.meta_data.drop(pos_indices_to_remove)
                else:
                    num_wanted_negative_class_intervals = (1/(1-negative_class_prec))*num_positive_class_intervals - num_positive_class_intervals
                    neg_indices = np.where(self.meta_data['label']==0.)[0]
                    neg_indices_to_remove = random.sample(neg_indices.tolist(), int(num_negative_class_intervals - num_wanted_negative_class_intervals))
                    neg_indices_to_remove = [self.meta_data.index[i] for i in neg_indices_to_remove]
                    self.meta_data = self.meta_data.drop(neg_indices_to_remove)

            if config['debug']:
                orig_size = len(self.meta_data)
                debug_size = int(config['debug_ratio'] * orig_size)
                self.meta_data = self.meta_data.sample(n=debug_size)
                print(f'debug mode, squeeze {d_type} data from {orig_size} to {debug_size}')

            if exp_dir:
                self.meta_data.to_csv(os.path.join(exp_dir,'dataframes', d_type+'_df.csv'), index=False)

            if clearml_task:
                report_df_to_clearml(self.meta_data, clearml_task, d_type)

            print('--------------------------------------------------------------')

    def __len__(self):
        return len(self.meta_data)
    
    def __getitem__(self, index):
        signal_path = self.meta_data.iloc[index]['interval_path']
        signal = np.load(os.path.join(self.folder_path,'intervals',signal_path))
        label = self.meta_data.iloc[index]['label']
        meta_data = self.meta_data.iloc[index]    
        signal = signal.reshape((1, len(signal)))
        if self.transform:
            signal = self.transform(signal)

        return (signal, label), meta_data.to_dict()
    

class AF_mixed_dataset(Dataset):
    def __init__(self, real_data_folder_path, fake_data_folder_path, clearml_task = False, exp_dir = None, transform = False, config=None, d_type='No data type specified'):
        super().__init__()

        self.transform = transform
        self.real_data_path = real_data_folder_path
        self.fake_data_path = fake_data_folder_path

        # Load real and fake data csv file from dataset folder:
        real_df = pd.read_csv(os.path.join(config['real_data_df_path'], d_type + '_df.csv'))
        real_df = drop_unnamed_columns(real_df)
        real_df['fake'] = 0
        
        # Add fake data only to training set
        if d_type == 'Train':
            fake_df = pd.read_csv(os.path.join(self.fake_data_path, 'meta_data.csv'))
            fake_df = drop_unnamed_columns(fake_df)
            fake_df['fake'] = 1

            # Split the training data into different amounts of real vs fake
            tot_pathology_train_amount = len(real_df[real_df['label']==1])
            real_pathology_amount = int(((100 - train_fake_perc) / 100) * tot_pathology_train_amount)
            fake__pathology_amount = int((train_fake_perc / 100) * tot_pathology_train_amount)
            real_normal_df = real_df[real_df['label']==0]
            real_pathology_df = real_df[real_df['label']==1].sample(n=real_pathology_amount)
            fake_df = fake_df.sample(n=fake__pathology_amount) 
            
            # Concatenate the real and fake DataFrames
            self.meta_data = pd.concat([real_normal_df, real_pathology_df, fake_df], ignore_index=True)
        
        else:
            self.meta_data = real_df.copy()


        if config['debug']:
                orig_size = len(self.meta_data)
                debug_size = int(config['debug_ratio'] * orig_size)
                self.meta_data = self.meta_data.sample(n=debug_size)
                print(f'debug mode, squeeze {d_type} data from {orig_size} to {debug_size}')


        label_stat = get_column_stats(self.meta_data, 'label')
        fake_stat = get_column_stats(self.meta_data, 'fake')

        print('--------------------------------------------------------------')
        print(f'created mixed {d_type} dataset with {len(self.meta_data)} intervals')
        print("\nLabel distribution:\n", label_stat)
        print("\nFake distribution:\n", fake_stat)
        print('--------------------------------------------------------------')

        if exp_dir:
            self.meta_data.to_csv(os.path.join(exp_dir,'dataframes', d_type+'_df.csv'), index=False)

        if clearml_task:
            report_df_to_clearml(self.meta_data, clearml_task, d_type)
            report_df_to_clearml(label_stat, clearml_task, d_type, title='label_stats')
            report_df_to_clearml(fake_stat, clearml_task, d_type,  title='fake_stats')
            

    def __len__(self):
        return len(self.meta_data)
    
    def __getitem__(self, index):
        signal_path = self.meta_data.iloc[index]['interval_path']
        if self.meta_data.iloc[index]['fake']:
            folder_path = self.fake_data_path
        else:
            folder_path = self.real_data_path
        signal = np.load(os.path.join(folder_path,'intervals',signal_path))
        label = self.meta_data.iloc[index]['label']
        meta_data = self.meta_data.iloc[index]    
        signal = signal.reshape((1, signal.size))
        if self.transform:
            signal = self.transform(signal)

        return (signal, label), meta_data.to_dict()


class AF_mixed_dataset_from_df(Dataset):
    def __init__(self, meta_data_df, real_data_folder_path, fake_data_folder_path, clearml_task = False, exp_dir = None, transform = False, d_type='No data type specified'):
        super().__init__()

        self.transform = transform
        self.real_data_path = real_data_folder_path
        self.fake_data_path = fake_data_folder_path

        # Use the provided DataFrame that holds the metadata of a dataset created by AF_mixed_dataset experiment
        self.meta_data = meta_data_df.copy()

        label_stat = get_column_stats(self.meta_data, 'label')
        fake_stat = get_column_stats(self.meta_data, 'fake')

        print('--------------------------------------------------------------')
        print(f'Read mixed {d_type} dataset with {len(self.meta_data)} intervals')
        print("\nLabel distribution:\n", label_stat)
        print("\nFake distribution:\n", fake_stat)
        print('--------------------------------------------------------------')

        if clearml_task:
            report_df_to_clearml(self.meta_data, clearml_task, d_type)
            report_df_to_clearml(label_stat, clearml_task, d_type, title='label_stats')
            report_df_to_clearml(fake_stat, clearml_task, d_type,  title='fake_stats')


    def __len__(self):
        return len(self.meta_data)


    def __getitem__(self, index):
        signal_path = self.meta_data.iloc[index]['interval_path']
        if self.meta_data.iloc[index]['fake']:
            folder_path = self.fake_data_path
        else:
            folder_path = self.real_data_path
        signal = np.load(os.path.join(folder_path,'intervals',signal_path))
        label = self.meta_data.iloc[index]['label']
        meta_data = self.meta_data.iloc[index]    
        signal = signal.reshape((1, signal.size))
        if self.transform:
            signal = self.transform(signal)

        return (signal, label), meta_data.to_dict()


if __name__ == '__main__':
    folder_path = 'C:/Users/nogak/Desktop/MyMaster/YoachimsCourse/dataset_len30_overlab5_chan0/'
    record_names = []
    for file in os.listdir('C:/Users/nogak/Desktop/MyMaster/YoachimsCourse/files'):
        if file.endswith('.hea'):  # we find only the .hea files.
            record_names.append(file[:-4])  # we remove the extensions, keeping only the number itself.
    config = load_config('config.yaml')
    ds = AF_dataset(folder_path, record_names, config=config)
    dataset_meta_data = ds.meta_data
    fs = 250
    for i, idx in enumerate(np.random.randint(0, len(ds) , 6)):
        (signal, label), meta_data = ds[idx]
        plt.subplot(3, 2, i + 1)
        t = np.arange(0, signal.shape[-1]/fs, 1/fs)
        plt.plot(t , signal.T)
        # plt.xlabel('time[sec]')
        plt.title(f'Label = {label}')

    plt.show()