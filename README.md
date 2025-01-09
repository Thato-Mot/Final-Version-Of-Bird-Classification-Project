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

This project focused on evaluating various machine learning and deep learning models for classifying bird species based on audio recordings. Below is a comprehensive analysis of the performance of these models:

### 1. Baseline Performance on Imbalanced Data
The initial dataset exhibited significant class imbalances, with some bird species overrepresented while others were underrepresented. The models struggled to generalise effectively, particularly for minority classes.

- **Random Forest (RF):**
  - **Accuracy**: 72.3%
  - **Strengths**: Quick to train, interpretable, and robust to overfitting.
  - **Weaknesses**: Struggled with minority class representation, leading to lower recall for underrepresented species.

- **Support Vector Machine (SVM):**
  - **Accuracy**: 74.4%
  - **Strengths**: Handled high-dimensional features well, better at separating classes than RF.
  - **Weaknesses**: Computationally expensive, especially for larger datasets.

- **k-Nearest Neighbours (KNN):**
  - **Accuracy**: 68.7%
  - **Strengths**: Easy to understand and implement.
  - **Weaknesses**: Poor performance on noisy data, sensitive to the choice of `k`, and computationally intensive for large datasets.

---

### 2. Impact of Data Augmentation
Data augmentation techniques (e.g., pitch shifting, noise addition, and time stretching) improved model generalisation and performance. The augmented dataset balanced the class distribution and exposed models to more diverse acoustic conditions.

- Augmentation **increased accuracy by 5–10%** across all models.
- **Random Forest (RF)**: Benefited from reduced class imbalance, showing improved F1-scores for minority classes.
- **SVM and KNN**: Demonstrated more stable performance post-augmentation, particularly in noisy environments.

---

### 3. Deep Learning Performance
Deep learning models outperformed traditional algorithms, particularly on the augmented dataset. The hierarchical feature learning capability of CNNs and the pre-trained knowledge from transfer learning significantly boosted performance.

#### 3.1 Convolutional Neural Networks (CNNs)
- **Accuracy**: 82.5% (baseline)
- **Key Insights**:
  - Performed well on larger feature sets like Mel spectrograms.
  - Struggled slightly with overlapping bird calls in real-world soundscapes.

#### 3.2 Transfer Learning (VGG-16)
- **Accuracy**: 89.2%
- **Controlled Test Accuracy**: 86.5%
- **Key Insights**:
  - Leveraged pre-trained weights to achieve superior performance with minimal fine-tuning.
  - Generalised well to new and unseen data, outperforming all other models.
  - Demonstrated robust feature extraction, particularly for species with unique vocal patterns.

---

### 4. Model Comparison on Key Metrics
The models were evaluated using multiple metrics, including precision, recall, F1-score, and area under the ROC curve (AUC). Below is a summary of the results:

| **Model**           | **Accuracy** | **Precision** | **Recall** | **F1-Score** | **AUC**  |
|----------------------|--------------|---------------|------------|--------------|----------|
| Random Forest (RF)   | 72.3%        | 71.8%         | 70.4%      | 71.1%        | 0.81     |
| SVM                  | 74.4%        | 74.1%         | 72.6%      | 73.3%        | 0.84     |
| KNN                  | 68.7%        | 67.5%         | 65.8%      | 66.6%        | 0.78     |
| CNN                  | 82.5%        | 81.7%         | 80.4%      | 81.0%        | 0.91     |
| Transfer Learning    | **89.2%**    | **88.7%**     | **87.9%**  | **88.3%**    | **0.95** |

---

### 5. Insights from Evaluation on Real-World Conditions
To assess real-world performance, the models were tested on unannotated soundscape recordings. These tests revealed several challenges:
- **Noise Sensitivity**: Models struggled with recordings containing overlapping bird calls and environmental noise.
- **Minority Class Generalisation**: Despite data augmentation, underrepresented species like *Anas strepera* (9 samples) had lower recall rates.
- **Best Performer**: VGG-16 achieved the highest performance on real-world data, maintaining an accuracy of 83.4% and an F1-score of 82.1%.

---

### 6. Visualisation of Results
Performance visualisation provided insights into the model’s strengths and areas for improvement:
- **Confusion Matrix**: Highlighted misclassifications, particularly among species with similar vocal characteristics.
- **ROC Curves**: Showed excellent separability for most species, with AUC values exceeding 0.9 for CNN and VGG-16.
- **Feature Importance (RF)**: MFCCs and Mel spectrograms emerged as the most influential features.

---

### Conclusion
The results demonstrated that transfer learning with VGG-16 is the most effective approach for bird call classification, achieving the highest accuracy and robustness. Future improvements could include:
- Expanding the dataset for better representation of minority classes.
- Experimenting with Transformer-based models for improved sequence modelling.
- Developing noise-resistant algorithms to enhance real-world applicability.


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
