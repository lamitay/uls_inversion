import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.functional import softmax
from clearml import Logger
from tqdm import tqdm
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from metrics import Metrics
import pandas as pd

class Trainer:
    def __init__(self, model, exp_dir, train_loader, validation_loader, test_loader, optimizer, criterion, scheduler, device, clearml, debug, epochs, data_dir, early_stopping):
        self.model = model
        self.exp_dir = exp_dir
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.class_labels = ['covid', 'pneumonia', 'regular', 'viral']
        self.results_dir = os.path.join(exp_dir, 'results')
        self.logger = None
        self.data_dir = data_dir
        self.clearml = clearml
        self.debug = debug
        self.early_stopping = early_stopping
        
        DEBUG_EPOCHS = 3

        if self.debug:
            self.epochs = DEBUG_EPOCHS
        else:
            self.epochs = epochs
        
        if self.clearml:
            self.logger = Logger.current_logger()


    def save_model(self, epoch):
        model_name = f"epoch_{epoch}_model.pth"
        model_path = os.path.join(self.exp_dir, 'models', model_name)
        torch.save(self.model.state_dict(), model_path)
        print(f"Saved model from Epoch: {epoch}' at {model_path}")

    def load_model(self, ckpt=0, different_exp_dir = None):
        if different_exp_dir != None:
            model_dir = os.path.join(different_exp_dir, 'models')
        else:
            model_dir = os.path.join(self.exp_dir, 'models')
        if ckpt:
            assert os.path.isfile(ckpt), f"Checkpoint '{ckpt}' not found."
            model_path = ckpt
        else:
            model_files = [f for f in os.listdir(model_dir) if f.endswith('model.pth')]
            if not model_files:
                raise FileNotFoundError("No model files found in the specified directory.")
            latest_model_file = max(model_files, key=lambda f: os.path.getctime(os.path.join(model_dir, f)))
            model_path = os.path.join(model_dir, latest_model_file)

        self.model.load_state_dict(torch.load(model_path))
        print(f'Loaded model weights from {model_path}')


    def train(self):
        best_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in tqdm(range(self.epochs)):            
            self.model.train()
            total_train_loss = 0.0
            num_train_examples = 0  
            for inputs, targets in tqdm(self.train_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_train_loss += loss.item() * inputs.size(0)
                num_train_examples += inputs.size(0)

                loss.backward()
                self.optimizer.step()

            train_loss = total_train_loss / num_train_examples
            val_loss = self.evaluate('validation', epoch)

            # Print epoch results
            print(f'Train Epoch: {epoch}'
                f'\tAverage Train Loss: {train_loss:.4f}'
                f'\tAverage Validation Loss: {val_loss:.4f}')
            
            # ClearML logging
            if self.clearml:
                self.logger.report_scalar(title="Epoch Loss", series="Training Loss", value=train_loss, iteration=epoch)
                self.logger.report_scalar(title="Epoch Loss", series="Validation Loss", value=val_loss, iteration=epoch)
                self.logger.report_scalar(title="Learning Rate", series="lr", value=self.scheduler.get_last_lr()[0], iteration=epoch)

            if val_loss < best_loss:
                best_loss = val_loss
                epochs_without_improvement = 0
                self.save_model(epoch=epoch)

            else:
                epochs_without_improvement += 1

            self.scheduler.step(val_loss)

            if self.early_stopping > 0:
                if epochs_without_improvement >= self.early_stopping:
                    print(f"Early stopping criterion met at Epoch {epoch}. Training stopped.")
                    break

        self.save_model(epoch=epoch)



    def evaluate(self, data_type, epoch=0, ckpt=None, different_exp_dir = None):
        if data_type == 'validation':
            loader = self.validation_loader
        elif data_type == 'test':
            loader = self.test_loader
            self.load_model(ckpt, different_exp_dir) # Last epoch model
        
        results_dir = os.path.join(self.exp_dir, 'results')

        self.model.eval()
        total_eval_loss = 0.0
        num_examples = 0
        true_labels = []
        predicted_labels = []
        predicted_probas = []

        with torch.no_grad():
            for inputs, targets in tqdm(loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_eval_loss += loss.item() * inputs.size(0)
                num_examples += inputs.size(0)

                # Get predicted labels using argmax
                thresholded_predictions = outputs.argmax(dim=1).cpu().numpy()

                # Apply softmax to get predicted probabilities
                predicted_probabilities = softmax(outputs, dim=1).cpu().numpy()
                
                # Convert tensors to numpy arrays
                true_labels.extend(targets.cpu().numpy())
                predicted_labels.extend(thresholded_predictions)
                predicted_probas.extend(predicted_probabilities)

        # Calculate Loss
        eval_loss = total_eval_loss / num_examples
        
        # Convert lists to numpy arrays
        true_labels = np.array(true_labels)
        predicted_labels = np.array(predicted_labels)
        predicted_probas = np.array(predicted_probas)

        # Calculate metrics
        accuracy, confusion_mat, f1_score, precision, recall, auroc, avg_prec = Metrics.calculate_metrics(data_type, epoch, true_labels, predicted_labels, predicted_probas, self.clearml, results_dir)

        if data_type == 'test':
            # Plot and log confusion matrix
            Metrics.plot_and_log_confusion_matrix(confusion_mat, self.class_labels, self.logger, self.clearml, results_dir)

            # # Plot ROC curve and log it to ClearML
            # Metrics.plot_roc_curve(true_labels, predicted_probas, self.logger, self.clearml, results_dir)

            # # Plot PR curve and log it to ClearML
            # Metrics.plot_pr_curve(true_labels, predicted_probas, self.logger, self.clearml, results_dir)
            
            # print('Started saving mistake images')
            # # Save up to 100 images of the network mistakes
            # Metrics.save_mistakes_images(true_labels, predicted_labels, self.data_dir, results_dir)
            # print('Finished saving mistake images')
            
            # print('Started saving correct images')
            # # Save 15 images of the networks correct predictions
            # Metrics.save_correct_images(true_labels, predicted_labels, self.data_dir, results_dir)
            # print('Finished saving correct images')

        return eval_loss