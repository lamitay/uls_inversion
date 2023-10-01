import argparse
from diffusers import DiffusionPipeline
import torch
import os
from datetime import datetime
import numpy as np
from scipy.stats import entropy
from PIL import Image
from tqdm import tqdm
import random

def calculate_entropy(image):
    hist = np.histogram(image.flatten(), bins=256, range=[0,256])[0]
    hist = hist / hist.sum()
    return entropy(hist)

def calculate_std(image):
    return np.std(image)

def generate_image(pipe, num_inference_steps, seed):
    image = pipe(
        batch_size=1,
        generator=torch.manual_seed(seed),
        num_inference_steps=num_inference_steps
        ).images
    return image

def main(args):
    # Initialize experiment dir
    # Get current date and time
    current_time = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    
    # Define the output directory with the current date and time
    exp_out_dir = os.path.join(args.output_dir, f'inference_{args.num_samples}_samples_{args.model_name}_{current_time}')
    out_samples_dir = os.path.join(exp_out_dir, 'samples')
    # Create output directory
    os.makedirs(exp_out_dir, exist_ok=True)
    os.makedirs(out_samples_dir, exist_ok=True)

    # Initialize the pipeline
    model_path = os.path.join(args.models_dir, args.model_name)
    pipe = DiffusionPipeline.from_pretrained(model_path, safety_checker=None, use_safetensors=True).to(args.device)

    print(f'Start generating {args.num_samples} samples from model {args.model_name}')

    saved_count = 0
    for i in tqdm(range(args.num_samples)):
        # Generate a random seed for this epoch to create variety of samples
        seed = args.seed + i
        # Initialize stats
        image_std = 0
        image_entropy = 0
        generation_attempt = 0
        while (image_std <= args.std_thresh) and (image_entropy <= args.entropy_thresh):
            # seed = random.randint(0, 2**32 - 1)
            seed = seed + (generation_attempt * 1000)

            image = generate_image(pipe, args.num_inference_steps, seed)[0]
            image_np = np.array(image)
            image_std = calculate_std(image_np)
            image_entropy = calculate_entropy(image_np)
            if image_std > args.std_thresh and image_entropy > args.entropy_thresh:
                image_resized = image.resize((args.img_size, args.img_size), Image.Resampling.LANCZOS)
                image_resized.save(os.path.join(out_samples_dir, f"{saved_count}.png"))
                saved_count += 1
                print(f'Sample #{i}  generated succesfully!! STD={image_std}, Entropy={image_entropy}\n')
                break  # Exit the while loop

            else:
                generation_attempt += 1
                print(f'Sample #{i} generation attempt {generation_attempt} Failed. Retrying!!!\n')
                print(f'Sample #{i} STD={image_std}, Entropy={image_entropy}\n')
    
    
    print(f'Succesfully generated {saved_count} samples from model {args.model_name}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate ultrasound images.')
    parser.add_argument('--model_name', type=str, default='ddpm-viral_3_mix_images_uls-128_3000_epochs', help='Model Name')
    # parser.add_argument('--model_name', type=str, default='ddpm-viral_uls-128_2000_epochs', help='Model Name')
    parser.add_argument('--models_dir', type=str, default='/home/lamitay/vscode_projects/uls_inversion/textual_inversion/', help='Path to the model')
    parser.add_argument('--output_dir', type=str, default="/home/lamitay/uls_experiments/ddpm/", help='Output directory')
    parser.add_argument('--device', type=str, default="cuda", help='Device')
    parser.add_argument('--num_inference_steps', type=int, default=500, help='Number of inference steps')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    parser.add_argument('--std_thresh', type=float, default=30.0, help='Standard deviation threshold to generate good quality images')
    parser.add_argument('--entropy_thresh', type=float, default=4.7, help='Entropy threshold to generate good quality images')
    
    args = parser.parse_args()
    main(args)
