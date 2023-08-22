import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from clearml import Task
from utils import *
from trainer import Trainer
from model import EcgResNet34
from dataset import *
import random
from datetime import datetime
import pandas as pd
from torchvision import models
import timm
import os


def main(args):
    # Set all random seeds for reproducability
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)

    # Constants
    NUM_CLASSES = 4
    NUM_WORKERS = 4

    # Init all parsed arguments
    data_dir = args.data_dir
    exp_base_dir = args.exp_base_dir
    uls_df_name = args.uls_df_name
    exp_name = args.exp_name
    clearml = args.clearml
    batch_size = args.batch_size
    loss = args.loss
    optimizer = args.optimizer
    lr_scheduler = args.lr_scheduler
    lr = args.lr
    model_type = args.model_type
    pretrained = args.pretrained
    num_layers_to_fine_tune = args.num_layers_to_fine_tune

    # Initialize experiment name and dirs
    timestamp = datetime.now().strftime("%Y%m%d_%H_%M_%S")
    exp_full_name = f"{exp_name}_{timestamp}"
    exp_dir = build_exp_dirs(exp_base_dir, exp_full_name)
    if clearml:
        clearml_task = Task.init(project_name="uls_inversion/classifier", task_name=exp_full_name)
    else:
        clearml_task=0

    # Load the ultrasound DataFrame
    df = pd.read_csv(os.path.join(data_dir, uls_df_name), index_col=None)

    # Define the input size and ImageNet mean and std based on the model type
    input_size = 224
    imagenet_mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # Define custom preprocessing transform
    preprocess_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(*imagenet_mean_std),
    ])


    # Create custom datasets with the transform
    train_dataset = LungUltrasoundDataset(d_type='train', 
                                          dataframe=df, 
                                          exp_dir=exp_dir, 
                                          clearml=clearml, 
                                          transform=preprocess_transform, 
                                          debug_mode=False)
    val_dataset = LungUltrasoundDataset(d_type='val', 
                                        dataframe=df, 
                                        exp_dir=exp_dir, 
                                        clearml=clearml, 
                                        transform=preprocess_transform, 
                                        debug_mode=False)
    test_dataset = LungUltrasoundDataset(d_type='test', 
                                         dataframe=df, 
                                         exp_dir=exp_dir, 
                                         clearml=clearml, 
                                         transform=preprocess_transform, 
                                         debug_mode=False)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    # Model and optimizers config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model selection and configuration
    if model_type == 'resnet':
        model = models.resnet18(pretrained=pretrained)
        penultimate_layer = model.fc.in_features
        model.fc = nn.Linear(penultimate_layer, NUM_CLASSES)
    elif model_type == 'vit':
        model = models.vision_transformer.vit_small_patch16_224(pretrained=pretrained)
        penultimate_layer = model.head.in_features
        model.head = nn.Linear(penultimate_layer, NUM_CLASSES)
    elif model_type == 'efficient_net':
        model = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=NUM_CLASSES)
    else:
        print('Unrecognized model type')
        exit()

    # Fine-tuning specific layers (optional)
    if num_layers_to_fine_tune > 0:
        for param in model.parameters():
            param.requires_grad = False
        for param in list(model.parameters())[-num_layers_to_fine_tune:]:
            param.requires_grad = True
    
    print_model_summary(model, batch_size, device='cpu')
    
    # Set up final parts of experiment
    model.to(device)
    
    if optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    if loss == 'ce':    
        criterion = F.cross_entropy
    if lr_scheduler == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True)
    
    # Training
    trainer = Trainer(model, exp_dir, train_loader, val_loader, test_loader, optimizer, criterion, scheduler, device, config, clearml_task, data_folder_path)
    print('Started training!')
    trainer.train()
    print('Finished training, Started test set evaluation!')
    trainer.evaluate(data_type='test')
    print('Finished experiment!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classifier Trainer')
    parser.add_argument('--data_dir', type=str, default='/home/lamitay/vscode_projects/covid19_ultrasound/data/image_dataset', help='Path to the ultrasound data')
    parser.add_argument('--uls_df_name', type=str, default='lung_uls_data.csv', help='Ultrasound data name')
    parser.add_argument('--exp_base_dir', type=str, default='/home/lamitay/uls_experiments', help='Path to the ultrasound experiments directory')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--loss', type=str, default='ce', help='Loss function name, can be - ce')
    parser.add_argument('--model_type', type=str, default='resnet', help='Model type, can be - resnet/vit/efficient_net')
    parser.add_argument('--pretrained', type=bool, default=True, help='Use imagenet pretrained weights')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='Optimizer name, can be - AdamW')
    parser.add_argument('--lr_scheduler', type=str, default='ReduceLROnPlateau', help='LR scheduler name, can be - ReduceLROnPlateau')
    parser.add_argument('--lr', type=int, default=1e-4, help='Learning rate')
    parser.add_argument('--clearml', type=bool, default=True, help='Create and log experiment to clearml')
    parser.add_argument('--exp_name', type=str, default='uls_inv_clsfr', help='Current experiment name')
    parser.add_argument('--num_layers_to_fine_tune', type=int, default=-1, help='If greater than 0, fine tune only this amount of final layers, otherwise train all layers')
    args = parser.parse_args()

    main(args)
