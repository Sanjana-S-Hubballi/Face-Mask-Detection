# Face mask Detection using MobileNet

## Overview
This project aims to detect and classify images faces in am image as either masked or unmasked.

## Workflow
To detect faces, I used OpenCV’s SSD deep learning face detector (based on ResNet-10). The script takes raw images from a given folder and outputs cropped face images into two folders — one for faces with masks and one for without masks.
After getting the face region, it crops out the face, resizes it to 224×224 pixels, normalizes it, and passes it through my MobileNetV2 model which predicts whether that face has a mask on or not.

### Data Augmentation and Preprocessing:
Before training, faces are loaded with augmentation:
- Random flips
- Rotations
- Zoom and shear
- Brightness variations
- Width and height shifts

### Model Training with Transfer Learning:
A MobileNetV2 model pre-trained on ImageNet is used as the base.
- Lower layers are frozen to retain generic features.
- Deeper layers are fine-tuned to adapt to mask detection.
- A custom classification head with dropout and dense layers is added.
- Class imbalance is handled with weighted loss.
- Model is trained with early stopping, learning rate adjustment, and checkpoint saving.

## Model Evaluation and Prediction:
The trained model is evaluated on the validation data. Predictions are generated and compared with actual labels to measure performance using:
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)
- False Rejection Rate (FRR) — Proportion of masked faces incorrectly classified as unmasked
- False Acceptance Rate (FAR) — Proportion of unmasked faces incorrectly classified as masked
These metrics help assess the model’s effectiveness, especially in real-world scenarios where minimizing false acceptances and rejections is critical.

## Training results
| **Metric**             | **Value**  |
|------------------------|------------|
| **Training Accuracy**  | 88.11%     |
| **Validation Accuracy**| 88.59%     |
| **Training Loss**      | 34.03%    |
| **Validation Loss**    | 24.84%     |
| **Final Learning Rate**| 9e-7       |

## Confusion matrix

![Confusion matrix.png](https://github.com/user-attachments/assets/9c89053c-edeb-4e4f-adb5-adc0f4b4755a)
