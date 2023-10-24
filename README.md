*This is the repo for my final project in the graduate course "Machine Learning 4 Health" in the faculty of Data Science, Technion.*

# Augmenting Lung Ultrasound with Synthetic Data: A Novel Application of Textual Inversion and Denoising Diffusion Probabilistic Models

![Ultrasound Inversion](https://github.com/lamitay/uls_inversion/blob/main/uls_inversion1.png?raw=true)


### Abstract
*The application of machine learning in healthcare, particularly in medical imaging, faces challenges such as data scarcity, quality, and class imbalance. This study seeks to mitigate these challenges in the domain of lung ultrasound imaging by introducing a novel data augmentation approach that employs Denoising Diffusion Probabilistic Models (DDPMs) and textual inversion techniques. Utilizing the open-source COVID-19 lung ultrasound dataset, which exhibits extreme class imbalance, the study compares the performance metrics of two prevalent classifier architectures: ResNet50 and Vision Transformer (ViT). Initially, a baseline classifier was established, with ResNet50 outperforming the ViT model across all key performance indicators. Efforts to address class imbalance through weight adjustments in the loss function led to decreased performance, indicating that this strategy may not be well-suited for this particular application. Experiments with textual inversion for synthetic ultrasound image generation yielded interesting yet inconsistent and clinically irrelevant results, underscoring the method's complexity and unpredictability. Two distinct DDPM models were trained: one using only three images from the minority class ('DDPM 3 mixed images') and another using an entire sequence from the training set of the minority class ('DDPM all training images'). While the latter approach resulted in a decline in performance metrics, the former yielded a significant boost, particularly when using the ResNet50 model. This suggests that DDPMs can capture beneficial features for classification when trained on a diverse image set. In conclusion, this study not only pioneers a novel data augmentation technique in the realm of lung ultrasound imaging but also serves as a foundational blueprint for future research. The methodology enhances the robustness and reliability of machine learning models in medical imaging, while also offering promising avenues for synthetic data generation.*


## Usage
The code is comprised of 2 main sections:
1. Classifier - code used to train and evaluate the classification models.
2. Textual_inversion - code used to train and generate synthetic images using textual inversion and DDPMs.

## Data
The data in this project is based on the open-source [COVID-19 lung ultrasound dataset](https://github.com/jannisborn/covid19_ultrasound/tree/master/data).

## Acknowledgements
**Course staff:**
Professor Uri Shalit and Bar Eini-Porat


