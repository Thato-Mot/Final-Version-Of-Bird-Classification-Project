# Bird Call Classification Project 🎵🐦

This project investigates the application of machine learning techniques for the automated classification of bird species based on their vocalisations. It is aimed at supporting biodiversity monitoring and conservation efforts by leveraging bioacoustics and cutting-edge machine learning models.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)

---

## Overview

Bird species are excellent indicators of environmental health, and their classification through vocalisations offers a non-invasive method for biodiversity monitoring. This project explores various machine learning and deep learning techniques to identify bird species from audio recordings. It incorporates feature extraction, augmentation, and model evaluation to achieve accurate classification.

---

## Features

- **Feature Extraction**: 
  - Mel-Frequency Cepstral Coefficients (MFCCs)
  - Mel spectrograms
  - Chroma features
  - Constant-Q Transform (CQT)

- **Machine Learning Models**:
  - Random Forest
  - Support Vector Machine (SVM)
  - k-Nearest Neighbours (KNN)

- **Deep Learning Architectures**:
  - Convolutional Neural Networks (CNNs)
  - Transfer learning (VGG-16)

- **Data Augmentation**:
  - Pitch shifting
  - Noise addition
  - Time stretching

- **Evaluation Metrics**:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Area Under the ROC Curve (AUC)

---

## Technologies Used

- Python
- Librosa (audio processing)
- NumPy, Pandas (data manipulation)
- Scikit-learn (machine learning)
- TensorFlow/Keras (deep learning)
- Matplotlib, Seaborn (visualisation)

---

## Dataset

The project uses the **Western Mediterranean Wetland Birds (WMWB)** dataset, which includes:
- **879 recordings** from 20 bird species.
- Annotations with start and end times for each bird vocalisation.
- Audio preprocessing:
  - Downsampling to 22,050 Hz.
  - Mono conversion.

---

## Methodology

1. **Data Preprocessing**:
   - Resampling and normalisation.
   - Sliding window segmentation for uniform feature extraction.

2. **Feature Extraction**:
   - Generating MFCCs, Mel spectrograms, and Chroma features.

3. **Model Training**:
   - Training traditional ML models and CNNs.
   - Transfer learning using pre-trained VGG-16 for improved accuracy.

4. **Data Augmentation**:
   - Enhancing model generalisation with synthetic samples.

5. **Evaluation**:
   - Models were assessed using metrics like accuracy and F1-Score.
   - A train-test split (70/30) and intra-species validation were used.

---

## Results

- **Random Forest Accuracy**: 72.3%
- **SVM Accuracy**: 74.4%
- **Transfer Learning (VGG-16) Accuracy**: 89.2%

---

## Future Work

- Expand the dataset to include more bird species.
- Explore Transformer architectures for improved sequence modelling.
- Develop a real-time bird call monitoring system.
- Incorporate unsupervised learning for better handling of unlabelled data.

---

## Acknowledgements

- **Supervisor**: Dr Yaaseen Martin, University of Cape Town.
- **Dataset Contributors**: Western Mediterranean Wetland Birds (WMWB) research team.
- **Inspiration**: Conservation efforts for avian biodiversity and curiosity.

---
