import torch
from datetime import datetime
import os
import numpy as np
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
from torch.nn.functional import softmax



def build_exp_dirs(exp_base_path, exp_name):
    assert os.path.exists(exp_base_path), f"Invalid experiments base path: {exp_base_path}"

    exp_dir = os.path.join(exp_base_path, exp_name)

    assert not os.path.exists(exp_dir), f"Experiment directory already exists: {exp_dir}"

    os.makedirs(exp_dir)
    # os.makedirs(os.path.join(exp_dir, "data"))
    os.makedirs(os.path.join(exp_dir, "models"))
    os.makedirs(os.path.join(exp_dir, "results"))
    os.makedirs(os.path.join(exp_dir, "dataframes"))

    return exp_dir

def print_model_summary(model, batch_size, num_ch=3, image_size=(224, 224), device='cpu'):
    """
    Prints the model summary for ultrasound images.

    Args:
        model (torch.nn.Module): The PyTorch model to summarize.
        batch_size (int): The batch size used for computing FLOPs.
        num_ch (int, optional): The number of input channels (e.g., 3 for RGB). Defaults to 3.
        image_size (tuple, optional): The height and width of the input images. Defaults to (224, 224).
        device (string, optional): The device to run the computation on. Defaults to 'cpu'.
    """
    summary(model.to(device), input_size=(num_ch, *image_size), device=device)
    # Create a sample input tensor
    input_size = (batch_size, num_ch, *image_size)
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
    '''
    Reports a dataframe as a table to clearml
    '''
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
    '''
    Gets a dataframe and returns statistics about the column (count and percentage)
    '''
    col_counts = curr_df[col_name].value_counts().reset_index()
    col_counts.columns = [col_name, 'Count']
    col_counts['Count'] = col_counts['Count'].astype(int)

    col_prec = curr_df[col_name].value_counts(normalize=True).reset_index()
    col_prec.columns = [col_name, 'Percentage']
    col_prec['Percentage'] = col_prec['Percentage'].round(4) * 100

    col_stat = pd.merge(col_counts, col_prec, on=col_name)

    return col_stat


def load_latest_model(model, path):
    list_of_files = glob.glob(os.path.join(path, 'models', 'epoch_*_model.pth')) 
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Loading model: {latest_file}")
    model.load_state_dict(torch.load(latest_file))
    return model



def create_and_save_embeddings(model, model_type, data_loader, embeddings_dir, d_type, device, clsfr_th=0.75):
    '''
    This function creates and saves embeddings using PCA and t-SNE for a given model, d_type and data loader.
    The outputs are saved npy, csv, png and html files of the embeddings.
    '''
    embeddings = []
    labels = []
    preds = []
    fakes = []
    paths = []

    # Register the hook based on the model type
    if model_type == 'resnet50':
        layer = model.layer4  # The penultimate layer for ResNet50
    elif model_type == 'vit':
        layer = model.blocks[-1]  # The penultimate layer for ViT
    elif model_type == 'efficient_net':
        layer = model._blocks[-1]  # The penultimate layer for EfficientNet

    def hook(module, input, output):
        output = output.detach().cpu().numpy()
        embeddings.append(output)

    handle = layer.register_forward_hook(hook)

    with torch.no_grad():
        for (inputs, targets), meta_data in tqdm(data_loader, desc=f"Processing {d_type} set"):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            
            # Get predicted labels using argmax
            thresholded_predictions = outputs.argmax(dim=1).cpu().numpy()

            # Apply softmax to get predicted probabilities
            predicted_probabilities = softmax(outputs, dim=1).cpu().numpy()         

            labels.append(targets.cpu().numpy())
            preds.append(thresholded_predictions.astype(int))
            fakes.append(meta_data['fake'])
            paths.append(meta_data['image_path'])
    
    handle.remove() # Unregister the hook

    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)
    preds = np.concatenate(preds)
    fakes = np.concatenate(fakes)
    paths = np.concatenate(paths)

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
    df['image_path'] = paths

    # Save the DataFrame
    df.to_csv(os.path.join(embeddings_dir, f'{d_type}_embeddings_reduced_PCA.csv'), index=False)

    # Plot PCA
    fig1 = px.scatter(df, x='component1', y='component2',
                      symbol=df['fake'].map({0: "cross", 1: "circle"}),  # Different symbols for 'fake' status
                      color='label',  # Coloring according to the label values
                      hover_data=['label', 'prediction', 'fake', 'image_path'])

    # Save the figure as an HTML file
    pio.write_html(fig1, os.path.join(embeddings_dir, f'label_embeddings_{d_type}_PCA_2d.html'))
   
    # Save the figure as png
    fig1.write_image(os.path.join(embeddings_dir, f'label_embeddings_{d_type}_PCA_2d.png'))

    # Plot the fake scatter only for the training set where there are fake samples
    if d_type != "Test":
        fig2 = px.scatter(df, x='component1', y='component2',
                          symbol=df['fake'].map({0: "cross", 1: "circle"}),  # Different symbols for 'fake' status
                          color='fake',  # Coloring according to the label values
                          hover_data=['label', 'prediction', 'fake', 'image_path'])

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
    df_tsne['image_path'] = paths

    # Save the DataFrame
    df_tsne.to_csv(os.path.join(embeddings_dir, f'{d_type}_embeddings_reduced_tSNE.csv'), index=False)

    fig3 = px.scatter(df_tsne, x='component1', y='component2',
                      symbol=df_tsne['fake'].map({0: "cross", 1: "circle"}),  # Different symbols for 'fake' status
                      color='label',  # Coloring according to the label values
                      hover_data=['label', 'prediction', 'fake', 'image_path'])

    pio.write_html(fig3, os.path.join(embeddings_dir, f'label_embeddings_{d_type}_tSNE_2d.html'))
    fig3.write_image(os.path.join(embeddings_dir, f'label_embeddings_{d_type}_tSNE_2d.png'))

    if d_type != "Test":
        fig4 = px.scatter(df_tsne, x='component1', y='component2',
                          symbol=df_tsne['fake'].map({0: "cross", 1: "circle"}),  # Different symbols for 'fake' status
                          color='fake',  # Coloring according to the label values
                          hover_data=['label', 'prediction', 'fake', 'image'])

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
