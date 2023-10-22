import pandas as pd
import matplotlib as plt
import seaborn as sns

def generate_plots_with_bottom_legend():
    # Data Preparation
    data = {
        'Experiment': ['Baseline', 'Baseline', 'Class weights', 'Class weights', 'DDPM 3 mixed', 'DDPM 3 mixed', 'DDPM all viral', 'DDPM all viral'],
        'Model': ['ViT', 'ResNet50', 'ViT', 'ResNet50', 'ViT', 'ResNet50', 'ViT', 'ResNet50'],
        'Accuracy': [71.84, 76.57, 68.91, 75.90, 70.72, 78.82, 66.89, 70.72],
        'F1 Score': [70.83, 74.59, 67.45, 74.15, 71.07, 76.80, 65.38, 69.06],
        'Precision': [71.02, 75.23, 66.36, 74.27, 74.14, 76.66, 68.19, 67.80]
    }
    df = pd.DataFrame(data)
    
    # Melt the DataFrame for seaborn
    metrics = ['Accuracy', 'F1 Score', 'Precision']
    df_melted = df.melt(id_vars=['Experiment', 'Model'], value_vars=metrics, var_name='Metric', value_name='Value')
    
    # Custom color palette for the models
    custom_palette = {'ViT': 'orange', 'ResNet50': 'blue'}
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Generate bar plots for each metric
    handles = []  # To store handles for the legend
    for i, metric in enumerate(metrics):
        ax = axes[i]
        sns.barplot(x='Experiment', y='Value', hue='Model', data=df_melted[df_melted['Metric'] == metric], ax=ax, palette=custom_palette)
        ax.set_title(f"{metric} by Experiment")
        ax.set_xlabel('Experiment')
        ax.set_ylabel(f"{metric} Value")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)  # Rotate x-axis labels
        ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
        ax.legend().remove()  # Remove individual legends
    
    # Add a single legend for the entire plot at the bottom
    for model, color in custom_palette.items():
        handles.append(plt.Rectangle((0,0),1,1, color=color, label=model))
    fig.legend(handles=handles, title='Model', loc='lower center', ncol=len(custom_palette), bbox_to_anchor=(0.5, -0.2))
    
    plt.tight_layout(rect=[0, 0.1, 1, 1])  # Adjust layout to make room for the legend
    plt.show()

# Run the function to generate the plots with a single bottom legend
generate_plots_with_bottom_legend()
