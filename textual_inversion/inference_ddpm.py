from diffusers import DiffusionPipeline
from diffusers.utils import make_image_grid
import torch 
import os
from datetime import datetime

# Get current date and time
current_time = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")

current_exp = 'ddpm-viral_3_mix_images_uls-128_3000_epochs'

# Define the output directory with the current date and time
output_dir = f'/home/lamitay/uls_experiments/ddpm/inference_{current_exp}_{current_time}/'

# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# repo_id = "runwayml/stable-diffusion-v1-5"
repo_id = f'/home/lamitay/vscode_projects/uls_inversion/textual_inversion/{current_exp}'
stable_diffusion = DiffusionPipeline.from_pretrained(repo_id, safety_checker=None, use_safetensors=True)
seed = 42
device = 'cuda'
pipe = stable_diffusion.to(device)
num_inference_steps = 1000
run = 3
# Sample some images from random noise (this is the backward diffusion process).
# The default pipeline output type is `List[PIL.Image]`
images = pipe(
    batch_size=16,
    generator=torch.manual_seed(seed),
    num_inference_steps=num_inference_steps
).images

# Make a grid out of the images
image_grid = make_image_grid(images, rows=4, cols=4)

# Save the images
test_dir = os.path.join(output_dir, "samples")
os.makedirs(test_dir, exist_ok=True)
image_grid.save(f"{test_dir}/sample_batch_{num_inference_steps}_steps_{run}.png")


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy
from PIL import Image

# Function to calculate entropy
def calculate_entropy(image):
    image_np = np.array(image)  # Convert PIL image to NumPy array
    hist = np.histogram(image_np.flatten(), bins=256, range=[0,256])[0]
    hist = hist / hist.sum()
    return entropy(hist)

# Function to calculate standard deviation
def calculate_std(image):
    image_np = np.array(image)  # Convert PIL image to NumPy array
    return np.std(image_np)


# Create a 4x4 grid of subplots
fig, axs = plt.subplots(4, 4, figsize=(15, 15))

# Flatten the axs array for easier indexing
axs = axs.flatten()

# Loop through each image and subplot
for i, (curr_image, ax) in enumerate(zip(images, axs)):
    # Calculate entropy and std
    ent = calculate_entropy(curr_image)
    std = calculate_std(curr_image)
    
    # Convert PIL image to NumPy array for histogram
    image_np = np.array(curr_image)
    
    # Plot histogram
    ax.hist(image_np.flatten(), bins=256, range=[0,256], color='gray')
    
    # Set title with entropy and std
    ax.set_title(f'Entropy: {ent:.2f}, Std: {std:.2f}')
    
    # Remove grid
    ax.grid(False)

# Show the plot
plt.tight_layout()
plt.show()
plt.savefig(f"{test_dir}/sample_batch_{num_inference_steps}_steps_{run}_histograms.png")
