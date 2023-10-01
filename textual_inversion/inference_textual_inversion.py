from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Get current date and time
current_time = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
# current_exp = 'textual_inversion_6_mixed_viral'
# current_exp = 'textual_inversion_viral_training'
current_exp = 'style_textual_inversion_6_mixed_viral'

ti_type = 'style'

# Define the output directory with the current date and time
output_dir = f'/home/lamitay/uls_experiments/textual_inversion/inference_{current_exp}_{current_time}/'

# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to load model and generate image
def generate_image(epoch_num, prompt):
    # Initialize the pipeline
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True).to("cuda")
    model_path = f'/home/lamitay/vscode_projects/uls_inversion/{current_exp}/learned_embeds-steps-{epoch_num}.safetensors'
    pipe.load_textual_inversion(model_path)
    image = pipe(prompt, num_inference_steps=50).images[0]
    return image

# 1. Subplots with samples from different epochs
epoch_nums = [500, 1000, 1500, 2500, 3000]
if ti_type == 'style':
    prompt = "A <viral-pneumonia-ultrasound> style ultrasound image"
else:
    prompt = "A <viral-pneumonia-ultrasound> ultrasound image"

fig, axs = plt.subplots(1, len(epoch_nums), figsize=(15, 5))  # Adjust figure size

for i, epoch_num in enumerate(epoch_nums):
    image = generate_image(epoch_num, prompt)
    axs[i].imshow(image)
    axs[i].set_title(f'Epoch {epoch_num}')
    axs[i].grid(False)  # Remove grid
    axs[i].axis('off')  # Remove axis

plt.tight_layout()  # Adjust layout to prevent overlap
plt.savefig(os.path.join(output_dir, 'model_development.png'))

# 2. Subplot of 2 samples with different prompts
if ti_type == 'style':
    prompts = ["A <viral-pneumonia-ultrasound> style ultrasound image", "A viral-pneumonia-ultraound style image"]
else:
    prompts = ["A <viral-pneumonia-ultrasound> ultrasound image", "A viral-pneumonia-ultraound image"]

fig, axs = plt.subplots(1, 2, figsize=(14, 5))  # Adjust figure size

for i, prompt in enumerate(prompts):
    image = generate_image(3000, prompt)  # Using the final model
    axs[i].imshow(image)
    axs[i].set_title(prompt)
    axs[i].grid(False)  # Remove grid
    axs[i].axis('off')  # Remove axis

plt.tight_layout()  # Adjust layout to prevent overlap
plt.savefig(os.path.join(output_dir, 'textual_inversion_success.png'))

# 3. Generate multiple images for different prompts
samples_output_dir = os.path.join(output_dir, 'samples')  # Replace with your desired output directory
num_images = 10  # Replace with your desired number of images

os.makedirs(samples_output_dir, exist_ok=True)

for i in range(num_images):
    if ti_type == 'style':
        image = generate_image(3000, "A <viral-pneumonia-ultrasound> style ultrasound image")  # Using the final model
    else:
        image = generate_image(3000, "A <viral-pneumonia-ultrasound> ultrasound image")  # Using the final model
    image.save(os.path.join(samples_output_dir, f'image_{i}.png'))
