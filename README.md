
# Autism Spectrum Disorder (ASD) Detection and Diagnosis System

## Overview

This repository contains the code and models developed for the early detection and confirmatory diagnosis of Autism Spectrum Disorder (ASD) using a combination of deep learning and machine learning techniques. The project is focused on providing a lightweight, easy-to-deploy system suitable for use in low-resource settings to assist in the early identification of autism in children.

## Table of Contents

- [Background](#background)
- [Problem Statement](#problem-statement)
- [Approach](#approach)
  - [Deep Learning for Preliminary Diagnosis](#deep-learning-for-preliminary-diagnosis)
  - [Machine Learning for Confirmatory Diagnosis](#machine-learning-for-confirmatory-diagnosis)
- [Data](#data)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [Ethical Considerations](#ethical-considerations)
- [Contributors](#contributors)
- [References](#references)

## Background

Autism Spectrum Disorder (ASD) is a complex neurodevelopmental condition characterized by persistent challenges in social interaction, communication, and restricted or repetitive activities. The global prevalence of ASD is approximately one in 160 individuals, with significant impacts on the quality of life for individuals with ASD and their families.

## Problem Statement

Traditional diagnostic approaches for ASD are often time-consuming, expensive, and can lead to delayed diagnosis, especially in low-resource settings. This project aims to address these challenges by developing a system that leverages machine learning and deep learning techniques for the efficient and accurate detection of ASD.

## Approach

### Deep Learning for Preliminary Diagnosis

We utilized the ResNet50 deep learning architecture to perform preliminary ASD detection based on facial image analysis. The model was trained on a dataset of 2,652 images, split into 80% training and 20% validation data. The model was fine-tuned with data augmentation techniques to enhance its generalizability.

### Machine Learning for Confirmatory Diagnosis

For the confirmatory diagnosis, an ensemble machine learning approach was adopted, combining Logistic Regression, Random Forest, and XGBoost classifiers. This ensemble was trained on the AQ-10 behavioral dataset, which includes demographic details and ASD-related behavioral traits.

## Data

- **Facial Image Dataset:** Used for training the ResNet50 model for preliminary ASD detection.
- **AQ-10 Behavioral Dataset:** Used for training the machine learning models for confirmatory diagnosis.

## Model Architecture

- **ResNet50:** Used for feature extraction from facial images.
- **Ensemble Classifier:** Combines Logistic Regression, Random Forest, and XGBoost for final classification based on behavioral data.

## Installation

To set up the project locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/asd-detection.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Training the Models:**
   - To train the deep learning model for preliminary diagnosis:
     ```bash
     python train_resnet50.py
     ```
   - To train the machine learning models for confirmatory diagnosis:
     ```bash
     python train_ensemble.py
     ```

2. **Making Predictions:**
   - For ASD detection using facial images:
     ```bash
     python predict_image.py --image path_to_image
     ```
   - For ASD confirmatory diagnosis using behavioral data:
     ```bash
     python predict_behavior.py --data path_to_data
     ```

## Results

- **ResNet50 Model:** Achieved an accuracy of 76.54% after 8 epochs of training.
- **Ensemble Classifier:** Achieved an accuracy of 92% on the test set, with strong performance metrics across all classes.

## Future Work

Future improvements include fine-tuning the models, expanding the dataset to include more diverse demographic data, and integrating explainable AI techniques to enhance the interpretability of the models.

## Ethical Considerations

The use of facial recognition and behavioral data raises significant ethical concerns, including issues of bias, privacy, and data security. This project emphasizes the need for transparency, fairness, and the inclusion of diverse stakeholders in the development process.

## Contributors

- [Your Name](https://github.com/your-username)

## References

The references used in this project include studies and datasets from various sources, such as WHO, Autism Speaks, Kaggle, and recent research papers on ASD prediction using machine learning.
