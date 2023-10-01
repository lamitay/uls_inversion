import os
import pandas as pd
from PIL import Image
import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy
from tqdm import tqdm


def calculate_entropy(image):
    hist = np.histogram(image.flatten(), bins=256, range=[0,256])[0]
    hist = hist / hist.sum()
    return entropy(hist)

def calculate_std(image):
    return np.std(image)

def df_stats(df, output_dir, model_name):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Descriptive statistics for 'std' and 'entropy'
    print("Statistics for 'std':")
    print(df['std'].describe())
    print("\nStatistics for 'entropy':")
    print(df['entropy'].describe())

    # Save statistics to a text file
    with open(os.path.join(output_dir, f'df_statistics_{model_name}.txt'), 'w') as f:
        f.write("Statistics for 'std':\n")
        f.write(str(df['std'].describe()))
        f.write("\n\nStatistics for 'entropy':\n")
        f.write(str(df['entropy'].describe()))

    # Histogram for 'std'
    plt.figure()
    plt.hist(df['std'], bins=20, alpha=0.5, color='g', label='std')
    plt.xlabel('Standard Deviation')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.title('Histogram of Standard Deviation')
    plt.savefig(os.path.join(output_dir, f'histogram_std_{model_name}.png'))
    # plt.show()

    # Histogram for 'entropy'
    plt.figure()
    plt.hist(df['entropy'], bins=20, alpha=0.5, color='b', label='entropy')
    plt.xlabel('Entropy')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.title('Histogram of Entropy')
    plt.savefig(os.path.join(output_dir, f'histogram_entropy_{model_name}.png'))
    # plt.show()

    # Boxplot for 'std'
    plt.figure()
    plt.boxplot(df['std'])
    plt.title('Boxplot of Standard Deviation')
    plt.savefig(os.path.join(output_dir, f'boxplot_std_{model_name}.png'))
    # plt.show()

    # Boxplot for 'entropy'
    plt.figure()
    plt.boxplot(df['entropy'])
    plt.title('Boxplot of Entropy')
    plt.savefig(os.path.join(output_dir, f'boxplot_entropy_{model_name}.png'))
    # plt.show()

def create_data_df(images_base_dir, diff_model_name):
    """
    Create a DataFrame containing image data, including file path, image stats and label.

    :param images_base_dir: Directory containing the images directory called 'samples'
    :return: DataFrame with image data
    """
    
    images_dir = images_base_dir
    
    # Initialize data and label dictionary
    data = []
    
    # Iterate through labels and images
    for image_name in tqdm(os.listdir(images_dir)):
        image_path = os.path.join(images_dir, image_name)
        image_name_strip = image_name.split('.')[0]
        label_name = 'viral'
        label = 3
        image = Image.open(image_path)
        image_np = np.array(image)
        image_std = calculate_std(image_np)
        image_entropy = calculate_entropy(image_np)
        image_width, image_height = image.size

        data.append([image_name_strip, image_path, diff_model_name, label_name, label, image_std, image_entropy, image_width, image_height])

    # Create DataFrame and add data_type based on custom splits
    df = pd.DataFrame(data, columns=['image_name', 'image_path', 'diff_model_name', 'label_name', 'label', 'std', 'entropy', 'width', 'height'])
    # df['image_name'] = df['image_name'].astype(int)
    # df = df.sort_values('image_name')

    return df

if __name__ == "__main__":
    # Argument parsing for command-line execution
    parser = argparse.ArgumentParser(description="Create a DataFrame from lung ultrasound data.")
    # parser.add_argument("--images_base_dir", default='/Users/amitaylev/Desktop/Amitay/Msc/4th semester/ML4Health/Final project/data/pocus_dataset/image_dataset/viral', help="Path to the directory containing the images.")
    # parser.add_argument("--diff_model_name", default="original_data", help="Name of the diffusion model used to generate the current dataset.")
    # parser.add_argument("--images_base_dir", default='/Users/amitaylev/Desktop/Amitay/Msc/4th semester/ML4Health/Final project/data/mixed_viral_data_3_images', help="Path to the directory containing the images.")
    # parser.add_argument("--diff_model_name", default="original_mixed_viral_data_3_images", help="Name of the diffusion model used to generate the current dataset.")
    parser.add_argument("--images_base_dir", default='/Users/amitaylev/Desktop/Amitay/Msc/4th semester/ML4Health/Final project/data/viral_training_data', help="Path to the directory containing the images.")
    parser.add_argument("--diff_model_name", default="original_viral_training_data", help="Name of the diffusion model used to generate the current dataset.")
    
    args = parser.parse_args()
    df = create_data_df(args.images_base_dir, args.diff_model_name)
    df_stats(df, os.path.join(args.images_base_dir, 'dataset_stats'), args.diff_model_name)

    # Save df
    output_path = os.path.join(args.images_base_dir, f'original_viral_uls_data_{args.diff_model_name}.csv')
    df.to_csv(output_path, index=False)
    print(f"DataFrame saved to {output_path}")
