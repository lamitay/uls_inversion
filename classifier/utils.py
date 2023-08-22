import torch
import wfdb
from datetime import datetime
import os
import numpy as np
from pecg import Preprocessing as Pre
from pecg.ecg import FiducialPoints as Fp
from wfdb import processing
import yaml
from torchsummary import summary
from fvcore.nn import flop_count, FlopCountAnalysis, flop_count_table
import pandas as pd
from clearml import Logger
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import random
import glob
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.io as pio
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def load_config(config_path):
    assert os.path.exists(config_path), f"Invalid config path: {config_path}"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def build_exp_dirs(exp_base_path, exp_name):
    assert os.path.exists(exp_base_path), f"Invalid experiments base path: {exp_base_path}"

    timestamp = datetime.now().strftime("%Y%m%d_%H_%M_%S")
    exp_dir = os.path.join(exp_base_path, f"{exp_name}_{timestamp}")

    assert not os.path.exists(exp_dir), f"Experiment directory already exists: {exp_dir}"

    os.makedirs(exp_dir)
    # os.makedirs(os.path.join(exp_dir, "data"))
    os.makedirs(os.path.join(exp_dir, "models"))
    os.makedirs(os.path.join(exp_dir, "results"))
    # os.makedirs(os.path.join(exp_dir, "logs"))
    os.makedirs(os.path.join(exp_dir, "dataframes"))

    return exp_dir


def get_record_names_from_folder(folder_path):
    assert os.path.exists(folder_path), f"Invalid folder path: {folder_path}"
    record_names = []
    for file in os.listdir(folder_path):
        if file.endswith('.hea'):  # we find only the .hea files.
            if file[:-4] == '00735' or file[:-4] == '03665':
                continue
            record_names.append(file[:-4])  # we remove the extensions, keeping only the number itself.
    return record_names

def split_records_according_to_class_dis(data_folder_path, wanted_ratio = 0.3, train_prec=80):
    meta_data = pd.read_csv(os.path.join(data_folder_path,'meta_data.csv'))
    record_names = np.unique(meta_data['record_file_name']).tolist()

    class_ratios = []
    for record in record_names:
        record_samples = meta_data[meta_data['record_file_name']==record]
        class_ratios.append(len(record_samples[record_samples['label']==1])/len(record_samples))
    record_df = pd.DataFrame({'record_names':record_names, 'ratio':class_ratios})
    
    delta = np.inf
    count = 0
    while delta > 0.07:
        record_names = list(record_df['record_names'])
        random.shuffle(record_names)
        train_records_names, val_records_names, test_records_names = split_records_train_val_test(record_names, train_prec)
        delta_train = np.abs(wanted_ratio - record_df[record_df['record_names'].isin(train_records_names)]['ratio'].mean())
        delta_val = np.abs(wanted_ratio - record_df[record_df['record_names'].isin(val_records_names)]['ratio'].mean())
        delta_test = np.abs(wanted_ratio - record_df[record_df['record_names'].isin(test_records_names)]['ratio'].mean())
        delta = (delta_train + delta_val + delta_test)/3
        count += 1

    print(f"Training set: {len(train_records_names)} records\nValidation set: {len(val_records_names)} records\nTest set: {len(test_records_names)} records")

    return train_records_names, val_records_names, test_records_names
    

def split_records_train_val_test(record_names, train_prec=80):
    num_of_train_val_records = round(len(record_names)*(train_prec/100))
    train_val_records_names = record_names[:num_of_train_val_records]
    test_records_names = record_names[num_of_train_val_records:]
    # Split train-validation to 80-20 % of the train_val
    num_of_train_records = round(len(train_val_records_names)*(train_prec/100))
    train_records_names = train_val_records_names[:num_of_train_records]
    val_records_names = train_val_records_names[num_of_train_records:]
    
    return train_records_names, val_records_names, test_records_names

def split_records_to_intervals_physioNet_challenge(record_df, sample_length):
    """split each of the records to intervals and the annotation to a binary 1D vector 

    Args:
        record (wfdb.record): the record object of wfdb
        annotation (wfdb.annotation): the annotation object of wfdb
        sample_length (float/int): the length of the required intervals in seconds 

    Returns:
        torch.Tensors: intervals, labels
    """
    file_name = os.path.join(folder_path, record_df['record_name'])
    mat_data = scipy.io.loadmat(file_name + ".mat")
    data = mat_data['val'] 
    fs = 300
    meta_data = {}
    meta_data['record_file_name'] = record_df['record_name']

    num_of_samples_in_interval = fs*sample_length
    num_of_intervals = int(np.floor(data.shape[1]/num_of_samples_in_interval))
    intervals = data[0, :num_of_intervals*num_of_samples_in_interval].reshape((num_of_intervals, num_of_samples_in_interval))
    if record_df['label'] == 'N':
        labels = np.zeros((1,num_of_intervals))
    else:
        labels = np.ones((1,num_of_intervals))
    bsqi_scores = []
    for i in range(num_of_intervals):
        bsqi_scores.append(bsqi(intervals[i,:], fs))

    meta_data['bsqi_scores'] = bsqi_scores
    meta_data['num_of_bit'] = np.zeros_like(labels[0])
    return intervals, labels[0], meta_data

def split_records_to_intervals(record, annotation, qrs, sample_length, channel, overlap, calc_bsqi):
    """split each of the records to intervals and the annotation to a binary 1D vector 

    Args:
        record (wfdb.record): the record object of wfdb
        annotation (wfdb.annotation): the annotation object of wfdb
        sample_length (float/int): the length of the required intervals in seconds 
        channel (int): if 0, take the first lead, if 1 take the second lead
        overlap (bool or float/int): if False, split without overlap. Else splitting the records with overlap (value in seconds!)

    Returns:
        torch.Tensors: intervals, labels
    """
    signal = torch.Tensor(record.p_signal[:, channel]) #start the signal from the index we got annotation for 
    fs = record.fs
    meta_data = {}
    meta_data['record_file_name'] = record.file_name[0]
    #create annotation signal with the length of records:
    annots_signal = torch.zeros_like(signal)
    for i, idx in enumerate(annotation.sample):
        if annotation.aux_note[i] == '(N':
            continue #label stays zero
        else:
            if i == len(annotation.sample)-1: #the last iteration in the loop:
                 annots_signal[idx:] = 1
            else:
                annots_signal[idx:annotation.sample[i+1]] = 1

    num_of_samples_in_interval = fs*sample_length
    if overlap == False: # split the record without overlap
        intervals, labels, meta_data['num_of_bit'], meta_data['bsqi_scores'] = segment_and_label(signal, annots_signal, qrs.sample, num_of_samples_in_interval, 0, fs, calc_bsqi=calc_bsqi)
    else:
        overlap_in_samples = overlap*fs
        intervals, labels, meta_data['num_of_bit'], meta_data['bsqi_scores'] = segment_and_label(signal, annots_signal, qrs.sample, num_of_samples_in_interval,  overlap_in_samples, fs, calc_bsqi=calc_bsqi)
    return intervals, labels, meta_data

def segment_and_label(x, y, qrs, m, w, fs, calc_bsqi=False):
    """
    Segments the time series x into segments of length m with overlap w,
    and creates a label vector for each segment based on the labels in y.

    Args:
    - x: a 1D PyTorch tensor representing the time series
    - y: a 1D PyTorch tensor representing the label vector
    - m: an integer representing the length of each segment
    - w: an integer representing the overlap between segments

    Returns:
    - segments: a 2D PyTorch tensor of shape (num_segments, m),
      where num_segments is the number of segments created from x
    - labels: a 1D PyTorch tensor of shape (num_segments,) representing
      the label vector for each segment
    """
    
    assert x.shape[0] == y.shape[0], 'length of time series should be equal to annotations signal'

    num_segments = (x.shape[0] - m) // (m - w) + 1  # compute number of segments
    segments = torch.zeros(num_segments, m)  # initialize segments tensor
    labels = torch.zeros(num_segments, dtype=torch.int64)  # initialize labels tensor
    num_of_bits = torch.zeros(num_segments, dtype=torch.int64)
    bsqi_scores = torch.zeros(num_segments, dtype=torch.float64)
    for i in range(num_segments):
        start = i * (m - w)
        end = start + m
        segment = x[start:end]
        label = y[start:end]
        if torch.all(label == 0):
            labels[i] = 0
            segments[i] = segment
        elif torch.all(label == 1):
            labels[i] = 1
            segments[i] = segment
        
        #calculate number of bits in segment:
        num_of_bits[i] = find_number_of_bits(start, end, qrs)

        if calc_bsqi:# Currently this calculation takes too many time so always false
            bsqi_scores[i] = bsqi(segment.numpy(), fs)

    # remove any segments that do not have a valid label
    final_segments = segments[segments.sum(axis=1)!=0, :]
    final_labels = labels[segments.sum(axis=1)!=0]
    num_of_bit = num_of_bits[segments.sum(axis=1)!=0]
    bsqi_scores = bsqi_scores[segments.sum(axis=1)!=0]

    return final_segments, final_labels, num_of_bit, bsqi_scores

def find_number_of_bits(i, j, qrs):
    start_index = np.searchsorted(qrs, i, side='left')
    end_index = np.searchsorted(qrs, j, side='right')
    return end_index - start_index

def bsqi(signal , fs):
    # First filter the signal using bandpass filter:
    pre = Pre.Preprocessing(signal, fs)
    filtered_signal = pre.bpfilt()
    fp = Fp.FiducialPoints(filtered_signal, fs)
    xqrs_inds = fp.xqrs()
    jqrs_inds = fp.jqrs()
    bsqi = pre.bsqi(xqrs_inds, jqrs_inds)   
    return bsqi

def create_dfs(segments, labels, meta_data):
    raw_data_df = pd.DataFrame(segments)
    labels_df = pd.DataFrame(labels)
    labels_df.columns = ['label']
    meta_data_df = pd.DataFrame(meta_data)
    return raw_data_df, labels_df, meta_data_df

def save_intervals_from_record(dataset_path, intervals, annots, meta_data, fs):
    record_file_name = meta_data['record_file_name']
    dfs = []
    for i in range(intervals.shape[0]):
        interval = intervals[i,:]
        label = annots[i]
        bsqi_score = meta_data['bsqi_scores'][i]
        file_name = f'{record_file_name[:-4]}_recordID_{i}_label_{label}_bsqi_{bsqi_score:.3f}.npy'

        # Save interval to npy file
        np.save(os.path.join(dataset_path, 'intervals', file_name), interval)

        # Save interval plot :
        t = np.arange(0, len(interval)/fs, 1/fs)
        plt.figure()
        plt.plot(t , interval)
        plt.xlabel('time[sec]')
        plt.title(f'Label = {label}')
        plt.savefig(os.path.join(dataset_path,'images', file_name[:-4]+'.png'))
        plt.close()
        
        interval_meta_data = {'record_file_name' : record_file_name,
                              'interval_path' : file_name,
                              'image_path' : file_name[:-4]+'.png',
                              'num_of_bits' : meta_data['num_of_bit'][i],
                              'bsqi_score' : bsqi_score,
                              'label' : label}
        
        dfs.append(pd.Series(interval_meta_data))
    record_meta_data = pd.concat(dfs,axis=1).T
    return record_meta_data

        
def create_dataset(folder_path, records_names, path_to_save_dataset, sample_length, channel, overlap, calc_bsqi = False):
    """create dataset folder from original 10 hours records

    Args:
        folder_path (string): path to records path
        records_names (list): records name to crate dataset with
        path_to_save_dataset (string): path to where dataset folder will be saved
        sample_length (int): length of each segment in seconds
        channel (int): whether to use the first channel (0) or the second (1)
        overlap (int): overlap between segments in seconds
    """
    path_to_save_dataset = os.path.join(path_to_save_dataset,f'dataset_len{sample_length}_overlab{overlap}_chan{channel}')
    # Check if dataset already exist:
    assert not os.path.exists(path_to_save_dataset), 'Dataset folder already exist, please remove exist folder'
    os.mkdir(path_to_save_dataset)
    os.mkdir(os.path.join(path_to_save_dataset, 'intervals'))
    os.mkdir(os.path.join(path_to_save_dataset, 'images'))
    meta_data_dfs = [] 
    for name in records_names:
        file_name = os.path.join(folder_path, name)
        record = wfdb.rdrecord(file_name)
        annotation = wfdb.rdann(file_name, 'atr')
        qrs = wfdb.rdann(file_name, 'qrs')
        intervals, annots, meta_data = split_records_to_intervals(record, 
                                                                  annotation, 
                                                                  qrs,
                                                                  sample_length = sample_length, #in seconds!
                                                                  channel = channel, # lead
                                                                  overlap = overlap, 
                                                                  calc_bsqi = calc_bsqi)
        
        record_meta_data = save_intervals_from_record(path_to_save_dataset, intervals, annots, meta_data, record.fs)
        meta_data_dfs.append(record_meta_data)
        print(f'Finish saving intervals of record {name}')
    # Save meta_data dataframe to csv file    
    pd.concat(meta_data_dfs, ignore_index=True).to_csv(os.path.join(path_to_save_dataset, 'meta_data.csv'))

def create_dataset_physioNet_challenge(folder_path, path_to_save_dataset, sample_length, channel, overlap, calc_bsqi = False):
    """create dataset folder from the phyionet challenge from 2017

    Args:
        folder_path (string): path to records path
        records_names (list): records name to crate dataset with
        path_to_save_dataset (string): path to where dataset folder will be saved
        sample_length (int): length of each segment in seconds
        channel (int): whether to use the first channel (0) or the second (1)
        overlap (int): overlap between segments in seconds
    """

    # Load the records name and filter to get only the Normal and AF records
    records_names = pd.read_csv(os.path.join(folder_path,'REFERENCE.csv'),names = ['record_name','label'])
    records_names = records_names[records_names['label'].isin(['A', 'N'])]
    path_to_save_dataset = os.path.join(path_to_save_dataset,f'physioNet_challenge_dataset_len{sample_length}_overlab{overlap}_chan{channel}')
    # Check if dataset already exist:
    if not os.path.exists(path_to_save_dataset):
        os.mkdir(path_to_save_dataset)
        os.mkdir(os.path.join(path_to_save_dataset, 'intervals'))
        os.mkdir(os.path.join(path_to_save_dataset, 'images'))
        
    meta_data_dfs = [] 
    for idx, row in records_names.iterrows():
        intervals, annots, meta_data = split_records_to_intervals_physioNet_challenge(row, sample_length = sample_length) #in seconds!
        
        record_meta_data = save_intervals_from_record(path_to_save_dataset, intervals, annots, meta_data, 300)
        meta_data_dfs.append(record_meta_data)
        print(f'Finish saving intervals of record {row["record_name"]}')
    # Save meta_data dataframe to csv file    
    pd.concat(meta_data_dfs, ignore_index=True).to_csv(os.path.join(path_to_save_dataset, 'meta_data.csv'))



def print_model_summary(model, batch_size, num_ch=1, samp_per_record=2500, device='cpu'):
    """
    Prints the model summary including the model architecture, number of parameters,
    and the total number of FLOPs (Floating Point Operations) required for a given input size.

    Args:
        model (torch.nn.Module): The PyTorch model to summarize.
        batch_size (int): The batch size used for computing FLOPs.
        num_ch (int, optional): The number of input channels. Defaults to 1.
        samp_per_record (int, optional): The number of samples per record. Defaults to 2500.
        device (string, optional):   

    Example:
        >>> model = MyModel()
        >>> batch_size = 64
        >>> num_ch = 1
        >>> samp_per_record = 2500
        >>> print_model_summary(model, batch_size, num_ch, samp_per_record)
        Output: Prints the model summary, parameter count, and total number of FLOPs.
    """
    summary(model.to(device), input_size=(num_ch, samp_per_record), device=device)
    # Create a sample input tensor
    input_size = (batch_size, num_ch, samp_per_record)
    rand_inputs = torch.randn(*input_size).to(device)
    # Compute FLOPs
    flops = FlopCountAnalysis(model, rand_inputs)
    print(flop_count_table(flops))
    print(f"Total number of FLOPs: {humanize_number(flops.total())} Flops")

def humanize_number(number):
    """
    Converts a large number into a human-readable format with appropriate unit suffixes.

    Args:
        number (float or int): The number to be formatted.

    Returns:
        str: The formatted number with unit suffix.

    Example:
        >>> number = 1512015806464
        >>> formatted_number = humanize_number(number)
        >>> print(formatted_number)
        Output: '1.51T'
    """

    units = ['', 'K', 'M', 'B', 'T']
    unit_index = 0
    while abs(number) >= 1000 and unit_index < len(units) - 1:
        number /= 1000.0
        unit_index += 1

    formatted_number = '{:.2f}{}'.format(number, units[unit_index])
    return formatted_number


def drop_unnamed_columns(df):
    """
    Drop columns starting with 'Unnamed' in a pandas DataFrame.
    
    Parameters:
        df (pandas.DataFrame): Input DataFrame.
        
    Returns:
        pandas.DataFrame: DataFrame with unnamed columns dropped.
        
    Raises:
        AssertionError: If no columns starting with 'Unnamed' are found.
    """
    
    columns = df.columns
    unnamed_columns = [col for col in columns if col.startswith('Unnamed')]
    if len(unnamed_columns) > 0:
        df = df.drop(unnamed_columns, axis=1)

    return df    


def report_df_to_clearml(df, d_type=None, title=None):
    df.index.name = "id"
    if title is None:
        title = f"{d_type}_data_table"
        sub_title = "Final data files"
    else:
        sub_title = title
    Logger.current_logger().report_table(
        f"{d_type}_{title}",
        sub_title, 
        iteration=0, 
        table_plot=df
    )


def get_column_stats(curr_df, col_name):
    col_counts = curr_df[col_name].value_counts().reset_index()
    col_counts.columns = [col_name, 'Count']
    col_counts['Count'] = col_counts['Count'].astype(int)

    col_prec = curr_df[col_name].value_counts(normalize=True).reset_index()
    col_prec.columns = [col_name, 'Percentage']
    col_prec['Percentage'] = col_prec['Percentage'].round(4) * 100

    col_stat = pd.merge(col_counts, col_prec, on=col_name)

    return col_stat

def print_dataset_distribution(dataset):
    labels = dataset.meta_data['label']
    print(f'label 0: {len(labels[labels==False])}   |   Prec: {"{:.2f}".format(len(labels[labels==False])/len(labels))}%') 
    print(f'label 1: {len(labels[labels==True])}   |   Prec: {"{:.2f}".format(len(labels[labels==True])/len(labels))}%')


def load_latest_model(model, path):
    list_of_files = glob.glob(os.path.join(path, 'models', 'epoch_*_model.pth')) 
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Loading model: {latest_file}")
    model.load_state_dict(torch.load(latest_file))
    return model


def extract_percentage(exp_name):
    """
    Function to extract the percentage value from the experiment name.

    Parameters:
    exp_name (str): The name of the experiment in the format 
    'tcml_Classifier_mixed_data_gen2_<percentage>_fake_percent_<date_time>'

    Returns:
    int: The percentage value in the experiment name.
    """
    return int(exp_name.split('_')[5])


def create_and_save_embeddings(model, data_loader, embeddings_dir, d_type, device, clsfr_th=0.75):
    '''
    This function creates and saves embeddings using PCA and t-SNE for a given model, d_type and data loader.
    The outputs are saved npy, csv, png and html files of the embeddings.
    '''
    embeddings = []
    labels = []
    preds = []
    fakes = []
    intervals = []

    with torch.no_grad():
        for (inputs, targets), meta_data in tqdm(data_loader, desc=f"Processing {d_type} set"):
            inputs = inputs.to(device).squeeze(1)
            targets = targets.to(device).float()

            embedding = model(inputs, return_embedding=True)
            outputs = model(inputs).squeeze(1)
            
            # Threshold the predictions
            thresholded_predictions = np.where(outputs.cpu().numpy() >= clsfr_th, 1, 0)        

            embeddings.append(embedding.cpu().numpy())
            labels.append(targets.cpu().numpy())
            preds.append(thresholded_predictions.astype(int))
            fakes.append(meta_data['fake'])
            intervals.append(meta_data['interval_path'])

    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)
    preds = np.concatenate(preds)
    fakes = np.concatenate(fakes)
    intervals = np.concatenate(intervals)

    # Save the embeddings
    np.save(os.path.join(embeddings_dir, f'{d_type}_embeddings.npy'), embeddings)

    # Reduce dimensionality using PCA - for distance metrics
    pca = PCA(n_components=2)
    embeddings_reduced = pca.fit_transform(embeddings)

    # Save the reduced embeddings
    np.save(os.path.join(embeddings_dir, f'{d_type}_embeddings_reduced_PCA.npy'), embeddings_reduced)
    
    # Create a DataFrame
    df = pd.DataFrame(embeddings_reduced, columns=['component1', 'component2'])
    df['label'] = labels
    df['prediction'] = preds
    df['fake'] = fakes
    df['interval_path'] = intervals

    # Save the DataFrame
    df.to_csv(os.path.join(embeddings_dir, f'{d_type}_embeddings_reduced_PCA.csv'), index=False)

    # Plot PCA
    fig1 = px.scatter(df, x='component1', y='component2',
                      symbol=df['fake'].map({0: "cross", 1: "circle"}),  # Different symbols for 'fake' status
                      color='label',  # Coloring according to the label values
                      hover_data=['label', 'prediction', 'fake', 'interval_path'])

    # Save the figure as an HTML file
    pio.write_html(fig1, os.path.join(embeddings_dir, f'label_embeddings_{d_type}_PCA_2d.html'))
   
    # Save the figure as png
    fig1.write_image(os.path.join(embeddings_dir, f'label_embeddings_{d_type}_PCA_2d.png'))

    # Plot the fake scatter only for the training set where there are fake samples
    if d_type != "Test":
        fig2 = px.scatter(df, x='component1', y='component2',
                          symbol=df['fake'].map({0: "cross", 1: "circle"}),  # Different symbols for 'fake' status
                          color='fake',  # Coloring according to the label values
                          hover_data=['label', 'prediction', 'fake', 'interval_path'])

        # Save the figure as an HTML file
        pio.write_html(fig2, os.path.join(embeddings_dir, f'fake_embeddings_{d_type}_PCA_2d.html'))
        
        # Save figures as png
        fig2.write_image(os.path.join(embeddings_dir, f'fake_embeddings_{d_type}_PCA_2d.png'))

    # Repeat the process for t-SNE reduced data
    tsne = TSNE(n_components=2)
    embeddings_reduced_tsne = tsne.fit_transform(embeddings)

    # Save the reduced embeddings with t-SNE
    np.save(os.path.join(embeddings_dir, f'{d_type}_embeddings_reduced_tSNE.npy'), embeddings_reduced_tsne)

    df_tsne = pd.DataFrame(embeddings_reduced_tsne, columns=['component1', 'component2'])
    df_tsne['label'] = labels
    df_tsne['prediction'] = preds
    df_tsne['fake'] = fakes
    df_tsne['interval_path'] = intervals

    # Save the DataFrame
    df_tsne.to_csv(os.path.join(embeddings_dir, f'{d_type}_embeddings_reduced_tSNE.csv'), index=False)

    fig3 = px.scatter(df_tsne, x='component1', y='component2',
                      symbol=df_tsne['fake'].map({0: "cross", 1: "circle"}),  # Different symbols for 'fake' status
                      color='label',  # Coloring according to the label values
                      hover_data=['label', 'prediction', 'fake', 'interval_path'])

    pio.write_html(fig3, os.path.join(embeddings_dir, f'label_embeddings_{d_type}_tSNE_2d.html'))
    fig3.write_image(os.path.join(embeddings_dir, f'label_embeddings_{d_type}_tSNE_2d.png'))

    if d_type != "Test":
        fig4 = px.scatter(df_tsne, x='component1', y='component2',
                          symbol=df_tsne['fake'].map({0: "cross", 1: "circle"}),  # Different symbols for 'fake' status
                          color='fake',  # Coloring according to the label values
                          hover_data=['label', 'prediction', 'fake', 'interval_path'])

        pio.write_html(fig4, os.path.join(embeddings_dir, f'fake_embeddings_{d_type}_tSNE_2d.html'))

        # Save figures as png
        fig4.write_image(os.path.join(embeddings_dir, f'fake_embeddings_{d_type}_tSNE_2d.png'))


def plot_pca_from_df(reduced_embeddings_df_path, output_dir, d_type, fake_perc, only_AF=False):
    
    # Read df
    df = pd.read_csv(reduced_embeddings_df_path, index_col=False)
    
    # Plot PCA
    fig1 = px.scatter(df, x='component1', y='component2',
                      symbol=df['fake'].map({0: "cross", 1: "circle"}),  # Different symbols for 'fake' status
                      color='label',  # Coloring according to the label values
                      hover_data=['label', 'prediction', 'fake', 'interval_path'])
    fig1.update_layout(title_text=f'{fake_perc}%_{d_type}_label_embeddings_PCA')

    # Save the figure as an HTML file
    pio.write_html(fig1, os.path.join(output_dir, f'{fake_perc}%_{d_type}_label_embeddings_PCA_2d.html'))
   
    # Save the figure as png
    fig1.write_image(os.path.join(output_dir, f'{fake_perc}%_{d_type}_label_embeddings_PCA_2d.png'))


    # Plot the fake scatter only for the training set where there are fake samples
    if d_type != "Test":
        # if flag only_AF is true, reduce the df to only plot the AF class (label==1)
        if only_AF:
            df = df[df['label']==1]

        fig2 = px.scatter(df, x='component1', y='component2',
                          symbol=df['fake'].map({0: "cross", 1: "circle"}),  # Different symbols for 'fake' status
                          color='fake',  # Coloring according to the label values
                          hover_data=['label', 'prediction', 'fake', 'interval_path'])
        fig2.update_layout(title_text=f'{fake_perc}%_{d_type}_AF_class_fake_embeddings_PCA')


        # Save the figure as an HTML file
        pio.write_html(fig2, os.path.join(output_dir, f'{fake_perc}%_{d_type}_AF_class_fake_embeddings_PCA_2d.html'))
        
        # Save figures as png
        fig2.write_image(os.path.join(output_dir, f'{fake_perc}%_{d_type}_AF_class_fake_embeddings_PCA_2d.png'))

    return fig1, fig2


def plot_tsne_from_df(reduced_embeddings_df_path, output_dir, d_type, fake_perc, only_AF=False):
    
    # Read df
    df = pd.read_csv(reduced_embeddings_df_path, index_col=False)

    # Plot PCA
    fig1 = px.scatter(df, x='component1', y='component2',
                      symbol=df['fake'].map({0: "cross", 1: "circle"}),  # Different symbols for 'fake' status
                      color='label',  # Coloring according to the label values
                      hover_data=['label', 'prediction', 'fake', 'interval_path'])
    fig1.update_layout(title_text=f'{fake_perc}%_{d_type}_label_embeddings_t-SNE')

    # Save the figure as an HTML file
    pio.write_html(fig1, os.path.join(output_dir, f'{fake_perc}%_{d_type}_label_embeddings_t-SNE_2d.html'))
   
    # Save the figure as png
    fig1.write_image(os.path.join(output_dir, f'{fake_perc}%_{d_type}_label_embeddings_t-SNE_2d.png'))

    # Plot the fake scatter only for the training set where there are fake samples
    if d_type != "Test":            
        # if flag only_AF is true, reduce the df to only plot the AF class (label==1)
        if only_AF:
            df = df[df['label']==1]

        fig2 = px.scatter(df, x='component1', y='component2',
                          symbol=df['fake'].map({0: "cross", 1: "circle"}),  # Different symbols for 'fake' status
                          color='fake',  # Coloring according to the label values
                          hover_data=['label', 'prediction', 'fake', 'interval_path'])
        fig2.update_layout(title_text=f'{fake_perc}%_{d_type}_AF_class_fake_embeddings_t-SNE')

        # Save the figure as an HTML file
        pio.write_html(fig2, os.path.join(output_dir, f'{fake_perc}%_{d_type}_AF_class_fake_embeddings_t-SNE_2d.html'))
        
        # Save figures as png
        fig2.write_image(os.path.join(output_dir, f'{fake_perc}%_{d_type}_AF_class_fake_embeddings_t-SNE_2d.png'))

    return fig1, fig2


import matplotlib.pyplot as plt

def plot_pca_from_df_mpl(reduced_embeddings_df_path, output_dir, d_type, fake_perc, only_AF=False):
    """
    Function to read PCA reduced embeddings from a csv file and plot them.

    Parameters:
    - reduced_embeddings_df_path: str, path to the csv file containing PCA reduced embeddings.
    - output_dir: str, directory path where the plot will be saved.
    - d_type: str, type of data (Training or Testing).
    - fake_perc: int, percentage of fake data present in the dataset.
    - only_AF: bool, if True, only plots the AF class.

    Returns:
    - fig: matplotlib figure object.
    - ax: matplotlib axis object.
    """
    
    # Read df
    df = pd.read_csv(reduced_embeddings_df_path, index_col=False)

    # Plot the fake scatter only for the training set where there are fake samples
    if d_type != "Test":
        # if flag only_AF is true, reduce the df to only plot the AF class (label==1)
        if only_AF:
            df = df[df['label'] == 1]

        # Create a matplotlib figure
        fig, ax = plt.subplots()

        # Plot real data
        ax.scatter(df[df['fake']==0]['component1'], df[df['fake']==0]['component2'], color='blue', marker='o', s=10, label='Real')
        
        # Plot fake data
        ax.scatter(df[df['fake']==1]['component1'], df[df['fake']==1]['component2'], color='orange', marker='o', s=10, label='Fake')
        
        ax.set_title(f'{fake_perc}%_{d_type}_AF_class_fake_embeddings_PCA')

        # Add a legend
        ax.legend()

        # Save the figure as png
        fig.savefig(os.path.join(output_dir, f'{fake_perc}%_{d_type}_AF_class_fake_embeddings_PCA_2d.png'))


    return fig, ax


def plot_tsne_from_df_mpl(reduced_embeddings_df_path, output_dir, d_type, fake_perc, only_AF=False):
    """
    Function to read t-SNE reduced embeddings from a csv file and plot them.

    Parameters:
    - reduced_embeddings_df_path: str, path to the csv file containing t-SNE reduced embeddings.
    - output_dir: str, directory path where the plot will be saved.
    - d_type: str, type of data (Training or Testing).
    - fake_perc: int, percentage of fake data present in the dataset.
    - only_AF: bool, if True, only plots the AF class.

    Returns:
    - fig: matplotlib figure object.
    - ax: matplotlib axis object.
    """
    
    # Read df
    df = pd.read_csv(reduced_embeddings_df_path, index_col=False)

    # Plot the fake scatter only for the training set where there are fake samples
    if d_type != "Test":            
        # if flag only_AF is true, reduce the df to only plot the AF class (label==1)
        if only_AF:
            df = df[df['label'] == 1]
            
        # Create a matplotlib figure
        fig, ax = plt.subplots()

                # Plot real data
        ax.scatter(df[df['fake']==0]['component1'], df[df['fake']==0]['component2'], color='blue', marker='o', s=10, label='Real')
        
        # Plot fake data
        ax.scatter(df[df['fake']==1]['component1'], df[df['fake']==1]['component2'], color='orange', marker='o', s=10, label='Fake')
        
        ax.set_title(f'{fake_perc}%_{d_type}_AF_class_fake_embeddings_t-SNE')

        # Add a legend
        ax.legend()
        
        # Save the figure as png
        fig.savefig(os.path.join(output_dir, f'{fake_perc}%_{d_type}_AF_class_fake_embeddings_t-SNE_2d.png'))

    return fig, ax


def create_subplot_grid(fig_axes, num_rows, num_cols, output_dir, reduce_met):
    """
    Function to create a grid of sub-plots for PCA or t-SNE reduced embeddings.

    Parameters:
    - fig_axes: list, each element is a tuple containing a matplotlib figure object and its associated axis.
    - num_rows: int, number of rows in the grid.
    - num_cols: int, number of columns in the grid.
    - output_dir: str, directory path where the plot will be saved.
    - reduce_met: str, reduction method ('PCA' or 't-SNE').

    Returns:
    - fig: matplotlib figure object.
    - axs: matplotlib axis object.
    """
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 15))
    fig.suptitle(f'Fake percentage AF class training set {reduce_met} embeddings', fontsize=20)

    for i, ax in enumerate(axs.flatten()):
        # Skip unnecessary sub-plots
        if i >= len(fig_axes):
            ax.axis('off')  # Hide unnecessary sub-plots
            continue

        fig, sub_ax = fig_axes[i]
        # Get collections from sub-axes (sub_ax)
        for collection in sub_ax.collections:
            ax.scatter(*collection.get_offsets().T, color=collection.get_facecolor())
        
        ax.set_xticks([])  # Remove x-axis values
        ax.set_yticks([])  # Remove y-axis values
        ax.set_title(sub_ax.get_title())
        ax.legend(['Real', 'Fake'])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fake_percentage_AF_class_training_set_{reduce_met}_embeddings.png'))  # Save the figure
    # plt.show()

    return fig, axs


def get_file_paths(experiments_dir, experiments, filename):
    file_paths = [os.path.join(experiments_dir, exp, 'embedding_results', filename) for exp in experiments if os.path.isfile(os.path.join(experiments_dir, exp, 'embedding_results', filename))]
    return file_paths




if __name__ == '__main__':
    # Create external dataset:
    folder_path = '/tcmldrive/NogaK/ECG_classification/training2017'
    create_dataset_physioNet_challenge(folder_path=folder_path, path_to_save_dataset='/tcmldrive/NogaK/ECG_classification/', sample_length=6, channel=0, overlap=0,calc_bsqi=True)


    # folder_path = 'C:/Users/nogak/Desktop/MyMaster/YoachimsCourse/files/'
    # record_names = []
    # for file in os.listdir(folder_path):
    #     if file.endswith('.hea'):  # we find only the .hea files.
    #         record_names.append(file[:-4])  # we remove the extensions, keeping only the number itself.

    # create_dataset(folder_path, record_names, 'C:/Users/nogak/Desktop/MyMaster/YoachimsCourse', 30, 0, 5, calc_bsqi = True)
    # ## TESTS :
    # # test1 verify that no mixed labels intervals are being created:
    # x = torch.rand(100000)
    # y = torch.zeros(100000)
    # y[::2] = 1
    # interval, labels = segment_and_label(x, y, 100, 0)
    # assert len(interval) == 0, 'In this tests all intervals should have mixed labels so no intervals should be created'

    # # test2 verify that if there are no mixed labels, all the intervals are being created:
    # x = torch.rand(100000)
    # y = torch.zeros(100000)
    # interval, labels = segment_and_label(x, y, 100, 0)
    # assert len(interval) == 100000/100, 'In this tests all intervals that possible needs to be created'