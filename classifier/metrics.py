import numpy as np
import matplotlib.pyplot as plt
from clearml import Logger
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report, roc_curve, auc, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, precision_recall_curve, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.preprocessing import label_binarize
import os
import seaborn as sns
import pandas as pd
import random

class Metrics:
    @staticmethod
    def calculate_metrics(d_type, epoch, true_labels, predicted_labels, probas, clearml=False, results_dir=None):
        # Calculate metrics
        accuracy = np.mean(true_labels == predicted_labels)
        confusion_mat = confusion_matrix(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='weighted')
        recall = recall_score(true_labels, predicted_labels, average='weighted')
        f1 = f1_score(true_labels, predicted_labels, average='weighted')

        # Convert true_labels to one-hot encoding
        n_classes = len((probas[0]))
        true_labels_one_hot = label_binarize(true_labels, classes=np.arange(n_classes))

        # Compute ROC AUC score
        # auroc = roc_auc_score(true_labels_one_hot, probas, multi_class='ovr')  # or multi_class='ovo' based on preference
        auroc = 0
        avg_prec = average_precision_score(true_labels_one_hot, probas, average='macro') # or 'micro', 'weighted' based on preference

        # Log metrics 
        Metrics.log_metric(d_type, epoch, 'Accuracy', accuracy, clearml, results_dir)
        Metrics.log_metric(d_type, epoch, 'F1 Score', f1, clearml, results_dir)
        Metrics.log_metric(d_type, epoch, 'Precision', precision, clearml, results_dir)
        Metrics.log_metric(d_type, epoch, 'Recall', recall, clearml, results_dir)
        Metrics.log_metric(d_type, epoch, 'AUROC', auroc, clearml, results_dir)
        Metrics.log_metric(d_type, epoch, 'Average Precision', avg_prec, clearml, results_dir)
        
        if d_type == 'test':
            # Create metrics table
            metrics_table = pd.DataFrame([
                ['Accuracy', accuracy],
                ['F1 Score', f1],
                ['Precision', precision],
                ['Recall', recall],
                ['AUROC', auroc],
                ['Average Precision', avg_prec]
                ], 
                columns=['Metric', 'Value'])
            Metrics.log_test_results(metrics_table, clearml, results_dir)

        return accuracy, confusion_mat, f1, precision, recall, auroc, avg_prec

    @staticmethod
    def log_test_results(metrics_table, log_to_clearml=False, results_dir=None):
        # Print and save metrics table
        print('Test set results:')
        print(metrics_table)

        with open(os.path.join(results_dir, 'metrics_results.txt'), 'a') as file:
            file.write(str(metrics_table) + '\n')

        if log_to_clearml:
            Logger.current_logger().report_table("Test set results", "Metrics", iteration=0, table_plot=metrics_table)

    @staticmethod
    def log_metric(d_type, epoch, metric_name, metric_value, log_to_clearml=False, results_dir=None):
        metric_name = f'{d_type}_{metric_name}'
        if log_to_clearml:
            Logger.current_logger().report_scalar(title=metric_name, series=metric_name, value=metric_value, iteration=epoch)

    @staticmethod
    def plot_and_log_confusion_matrix(confusion_mat, class_labels, task, log_to_clearml=False, results_dir=None):
        # Plot the confusion matrix
        plt.figure(figsize=(8, 6))        
        
        # Plot the confusion matrix using seaborn
        svm=sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', cbar=False)

        # Add title and labels
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')

        # Save confusion matrix as PNG
        confusion_matrix_file = os.path.join(results_dir, 'confusion_matrix.png')
        plt.savefig(confusion_matrix_file, dpi=400)

        # Show the plot
        plt.show()

        # Close the plot
        plt.close()

    @staticmethod
    def plot_roc_curve(true_labels, probas, task, log_to_clearml=False, results_dir=None):
        # Calculate false positive rate, true positive rate, and thresholds
        fpr, tpr, thresholds = roc_curve(true_labels, probas)

        # Calculate area under the ROC curve
        roc_auc = auc(fpr, tpr)
        
        # Create the ROC curve display
        roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)

        # Plot the ROC curve
        roc_display.plot()

        # Set the plot title
        plt.title('Receiver Operating Characteristic (AUC = {:.2f})'.format(roc_auc))

        # Save ROC curve as PNG
        roc_curve_file = os.path.join(results_dir, 'roc_curve.png')
        plt.savefig(roc_curve_file, dpi=400)

        # Show the plot
        plt.show()
        
        # Close the plot
        plt.close()

    @staticmethod
    def plot_pr_curve(true_labels, probas, task, log_to_clearml=False, results_dir=None):
        precision, recall, _ = precision_recall_curve(true_labels, probas)
        auprc = average_precision_score(true_labels, probas)
        
        # Create the Precision-Recall display
        pr_display = PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=auprc)

        # Plot the Precision-Recall curve
        pr_display.plot()

        # Set the plot title
        plt.title('Precision-Recall Curve (AUPRC = {:.2f})'.format(auprc))

        # Save PR curve as PNG
        pr_curve_file = os.path.join(results_dir, 'pr_curve.png')
        plt.savefig(pr_curve_file, dpi=400)

        # Show the plot
        plt.show()
        
        # Close the plot
        plt.close()

    @staticmethod
    def save_mistakes_images(true_labels, predicted_labels, meta_data, dataset_path, results_dir=None):
        # Add mistakes folder in the results_dir:
        os.mkdir(os.path.join(results_dir, 'mistakes'))

        # Add to meta data the predicted labels
        meta_data['prediction'] = predicted_labels
        # mistakes = true_labels != predicted_labels
        # mistakes_meta_data = meta_data[mistakes]

        # Filter FP and FN mistakes
        fp_mistakes = meta_data[(true_labels == 0) & (predicted_labels == 1)]
        fn_mistakes = meta_data[(true_labels == 1) & (predicted_labels == 0)]

        # Randomly select mistakes to include
        selected_mistakes = []
        selected_mistakes.extend(random.sample(list(fp_mistakes.iterrows()), min(50, len(fp_mistakes))))
        selected_mistakes.extend(random.sample(list(fn_mistakes.iterrows()), min(50, len(fn_mistakes))))

        # Save plots of selected mistakes
        for idx, mistake in selected_mistakes:
            signal = np.load(os.path.join(dataset_path, 'intervals', mistake['interval_path']))
            mistake_type = 'FP' if mistake['label'] == 0 else 'FN'
            file_name = mistake['image_path'][:-4] + f"_pred_{mistake['prediction']}_{mistake_type}.png"
            file_path = os.path.join(results_dir, 'mistakes', file_name)

            # Plot and save interval
            t = np.arange(0, len(signal) / 250, 1 / 250)
            plt.figure()
            plt.plot(t, signal)
            plt.xlabel('time[sec]')
            # plt.title(f"True Label = {mistake['label']}, Predicted Label = {mistake['prediction']}")
            plt.title(mistake_type + '_sample_' + mistake['image_path'][:-4] + f"_pred_{mistake['prediction']}")
            plt.savefig(file_path)
            plt.close()



    @staticmethod
    def save_correct_images(true_labels, predicted_labels, meta_data, dataset_path, results_dir=None):
        # Add mistakes folder in the results_dir:
        os.mkdir(os.path.join(results_dir, 'corrects'))
        # Add to meta data the predicted labels
        meta_data['prediction'] = predicted_labels
        correct = true_labels == predicted_labels
        correct_meta_data = meta_data.iloc[correct]
        if len(correct_meta_data) > 15: # save plots of maximum 15 correct
            correct_meta_data = correct_meta_data[:15]
        for idx, correct in correct_meta_data.iterrows():
            signal = np.load(os.path.join(dataset_path, 'intervals', correct['interval_path']))
            correct_type = 'TP' if correct['label'] == 1 else 'TN'
            # Save interval plot :
            t = np.arange(0, len(signal)/250, 1/250)
            plt.figure()
            plt.plot(t , signal)
            plt.xlabel('time[sec]')
            # plt.title(f"True Label = {correct['label']}, Predicted Label = {correct['prediction']}")
            plt.title(correct_type + '_sample_' + correct['image_path'][:-4] + f"_pred_{correct['prediction']}")
            plt.savefig(os.path.join(results_dir,'corrects',correct['image_path'][:-4]+f"_pred_{correct['prediction']}_{correct_type}.png"))
            plt.close()