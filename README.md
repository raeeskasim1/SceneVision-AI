# 🏞️ SceneVision-AI

A deep learning project for natural scene image classification using a custom Convolutional Neural Network (CNN) built with PyTorch.

## Project Overview

SceneVision-AI is an end-to-end computer vision project that classifies natural scene images into six categories using a custom CNN.

The project demonstrates the complete deep learning workflow, including:

- Dataset preparation
- Model training
- Validation
- Evaluation
- Prediction
- Performance visualization
- Feature map analysis
- Misclassified image analysis

The project was built as a professional portfolio project following machine learning engineering best practices.

## Features

- Custom CNN architecture
- GPU (CUDA) training
- Early Stopping
- Learning Rate Scheduler
- Model Checkpointing
- Single Image Prediction
- Confusion Matrix
- Training Curves
- Feature Map Visualization
- Misclassified Image Analysis

## Dataset

Dataset: Intel Image Classification Dataset

Classes:

- Buildings
- Forest
- Glacier
- Mountain
- Sea
- Street

Image Size: 150 × 150 RGB

### Project Structure
SceneVision-AI       
│           
├── assets/     
├── data/     
├── outputs/    
├── src/     
│   ├── model/cnn.py     
│   ├── dataset.py    
│   ├── train.py    
│   ├── evaluate.py    
│   ├── predict.py    
│   └── visualize.py     
│     
├── requirements.txt       
└── README.md     

## Model Architecture

The project started with a 3-block baseline CNN and was improved to a 4-block CNN through controlled experiments.

### Final Architecture

- 4 Convolution Blocks
- Batch Normalization
- ReLU Activation
- Max Pooling
- Fully Connected Classifier
- Dropout (0.5)


## Tech Stack

- Python
- PyTorch
- TorchVision
- NumPy
- Matplotlib
- Scikit-learn
- Pillow

### Training Configuration

- Custom CNN (4 Convolution Blocks)
- Batch Normalization
- ReLU Activation
- Max Pooling
- Adam Optimizer
- ReduceLROnPlateau Learning Rate Scheduler
- Early Stopping (Patience = 10)
- CrossEntropyLoss
- CUDA GPU Training

# Results

## 📊 Version 1 (Baseline)

### Performance

| Metric | Score |
|--------|------:|
| Train Accuracy | **93.36%** |
| Validation Accuracy | **86.96%** |
| Test Accuracy | **85.93%** |

---

## 🚀 Version 2 — Improved CNN

Added a fourth convolution block:

3 → 32 → 64 → 128 → 256

| Metric              | v1.0 | v2.0 |
| ------------------- | ---: | ---: |
| Train Accuracy      | 93.36% | 91.12% |
| Validation Accuracy | 86.96% | 87.10% |
| Test Accuracy       | 85.93% | **87.77%** |

**Improvement: +1.84 percentage points**

Detailed E1–E4 experiments are documented in [`experiments.md`](docs/experiments.md).

---

### Classification Report

| Class     | Precision | Recall | F1-Score |
| --------- | --------: | -----: | -------: |
| Buildings | 0.86 | 0.85 | 0.85 |
| Forest    | **0.97** | **0.98** | **0.97** |
| Glacier   | 0.82 | 0.86 | 0.84 |
| Mountain  | 0.84 | 0.82 | 0.83 |
| Sea       | 0.90 | 0.89 | 0.89 |
| Street    | 0.89 | 0.88 | 0.88 |

**Overall Accuracy:** **87.77%**

### Confusion Matrix

The confusion matrix provides a detailed view of the model's predictions across all scene categories.

![Confusion Matrix](assets/reports/confusion_matrix_e4.png)

---

### Key Observations

- Best-performing class: **Forest** (F1-score: **0.97**)
- Most challenging classes: **Glacier** and **Mountain**
- Mountain and Glacier remain challenging due to visually similar scene characteristics.
- The improved CNN increased test accuracy from **85.93% to 87.77%**.

---

## Training Curves

The training process is visualized using loss and accuracy curves.

- Training Loss vs Validation Loss
- Training Accuracy vs Validation Accuracy

![Training Curves](assets/curves/training_curves_e4.png)

## Feature Map Visualization

The feature maps below show how the CNN progressively transforms the input image into increasingly abstract representations.

### Input Image

![Input Image](assets/sample_prediction_image.jpg)

---

- Block 1: Edge and texture detection
- Block 2: Mid-level structures and patterns
- Block 3: High-level semantic representations for classification

### Block 1

![Block 1 Feature Maps](assets/feature_maps/feature_map1_v2.png)

---

### Block 2

![Block 2 Feature Maps](assets/feature_maps/feature_map2_v2.png)

---

### Block 3

![Block 3 Feature Maps](assets/feature_maps/feature_map3_v2.png)

---

### Block 4

![Block 4 Feature Maps](assets/feature_maps/feature_map4_v2.png)

---

As information flows through the CNN, the learned representations become increasingly abstract.

- Block 1 extracts low-level features such as edges and textures.
- Block 2 combines these into larger visual patterns.
- Block 3 focuses on high-level semantic features that help distinguish scene categories.

## Misclassified Images

Analyzing incorrect predictions helps identify the model's weaknesses and guides future improvements.

The visualization below shows randomly selected misclassified test images along with:

- Actual class
- Predicted class
- Prediction confidence

This analysis reveals which scene categories are visually similar and where the model struggles most.

![Misclassified Images](assets/misclassified_images_v2.png)

## Version History

| Version | Description |
| ------- | ----------- |
| v1.0 | Baseline custom CNN with complete training, evaluation, prediction, and visualization pipeline |
| v2.0 | Improved CNN with an additional convolution block, achieving **87.77% test accuracy** |