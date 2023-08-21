import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from clearml import Logger
from tqdm import tqdm
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from metrics import Metrics
import pandas as pd

class Trainer:
    def __init__(self, model, exp_dir, train_loader, validation_loader, test_loader, optimizer, criterion, scheduler, device, config, clearml_task, dataset_path):
        self.model = model
        self.exp_dir = exp_dir
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        self.scheduler = scheduler
        self.class_labels = ['Normal', 'AF']
        self.results_dir = os.path.join(exp_dir, 'results')
        self.logger = None
        self.dataset_path = dataset_path
        if self.config['debug']:
            self.epochs = self.config['debug_epochs']
        else:
            self.epochs = self.config['epochs']
        
        if self.config['clearml']:
            self.logger = Logger(clearml_task)


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
            for (inputs, targets), meta_data in tqdm(self.train_loader):
                inputs = inputs.to(self.device).squeeze(1)
                targets = targets.to(self.device).float()

                self.optimizer.zero_grad()
                outputs = self.model(inputs).squeeze()
                loss = self.criterion(outputs, targets.squeeze())

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
            if self.config['clearml']:
                self.logger.report_scalar(title="Epoch Loss", series="Training Loss", value=train_loss, iteration=epoch)
                self.logger.report_scalar(title="Epoch Loss", series="Validation Loss", value=val_loss, iteration=epoch)

            if val_loss < best_loss:
                best_loss = val_loss
                epochs_without_improvement = 0
                self.save_model(epoch=epoch)

            else:
                epochs_without_improvement += 1

            self.scheduler.step(val_loss)

            if self.config['early_stopping']:
                if epochs_without_improvement >= self.config['early_stopping_patience']:
                    # self.logger.report_text("Early stopping criterion met. Training stopped.")
                    print(f"Early stopping criterion met at Epoch {epoch}. Training stopped.")
                    break

        
        # TODO: Make sure this happens for early stopping as well
        self.save_model(epoch=epoch)



    def evaluate(self, data_type, epoch=0, ckpt=None, different_exp_dir = None):
        if data_type == 'validation':
            loader = self.validation_loader
        elif data_type == 'test':
            loader = self.test_loader
            self.load_model(ckpt, different_exp_dir)
        
        results_dir = os.path.join(self.exp_dir, 'results')

        self.model.eval()
        total_eval_loss = 0.0
        num_examples = 0
        true_labels = []
        predicted_labels = []
        predicted_probas = []
        meta_data_list = []
        meta_data_list = []

        with torch.no_grad():
            for (inputs, targets), meta_data in tqdm(loader):
                inputs = inputs.to(self.device).squeeze(1)
                targets = targets.to(self.device).float()

                outputs = self.model(inputs).squeeze(1)
                loss = self.criterion(outputs, targets)

                total_eval_loss += loss.item() * inputs.size(0)
                num_examples += inputs.size(0)

                # Threshold the predictions
                thresholded_predictions = np.where(outputs.cpu().numpy() >= self.config['classifier_th'], 1, 0)        

                # Convert tensors to numpy arrays
                true_labels.extend(targets.cpu().numpy())
                predicted_labels.extend(thresholded_predictions)
                predicted_probas.extend(outputs.cpu().numpy())
                meta_data_list.append(pd.DataFrame(meta_data))

        # Calculate Loss
        eval_loss = total_eval_loss / num_examples
        
        # Convert lists to numpy arrays
        true_labels = np.array(true_labels)
        predicted_labels = np.array(predicted_labels)
        predicted_probas = np.array(predicted_probas)
        meta_data_df = pd.concat(meta_data_list, axis=0, ignore_index=True)
        # Calculate metrics
        accuracy, confusion_mat, f1_score, precision, recall, auroc, avg_prec = Metrics.calculate_metrics(data_type, epoch, true_labels, predicted_labels, predicted_probas, self.config['clearml'], results_dir)

        if data_type == 'test':
            # Plot and log confusion matrix
            Metrics.plot_and_log_confusion_matrix(confusion_mat, self.class_labels, self.logger, self.config['clearml'], results_dir)

            # Plot ROC curve and log it to ClearML
            Metrics.plot_roc_curve(true_labels, predicted_probas, self.logger, self.config['clearml'], results_dir)

            # Plot PR curve and log it to ClearML
            Metrics.plot_pr_curve(true_labels, predicted_probas, self.logger, self.config['clearml'], results_dir)
            
            print('Started saving mistake images')
            # Save up to 100 images of the network mistakes
            Metrics.save_mistakes_images(true_labels, predicted_labels, meta_data_df, self.dataset_path, results_dir)
            print('Finished saving mistake images')
            
            print('Started saving correct images')
            # Save 15 images of the networks correct predictions
            Metrics.save_correct_images(true_labels, predicted_labels, meta_data_df, self.dataset_path, results_dir)
            print('Finished saving correct images')

        return eval_loss