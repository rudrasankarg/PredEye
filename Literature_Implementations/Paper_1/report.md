# Paper 1: Multimodal Appearance-Based Gaze-Controlled Virtual Keyboard

## Dataset
- **Name:** UnityEyes (Synthetic Dataset) + User Calibration Images
- **Size:** ~20,000 eye images
- **Details:** Images are resized to 100x100 in grayscale. Data augmentation (rotation, width/height shifting) was applied.

## Algorithms & Models
- **Model:** 4-layer Convolutional Neural Network (CNN)
- **Task:** 9-direction gaze classification (NW, N, NE, W, C, E, SW, S, SE).

## Implementation Details
- **Optimization Technique:** Adam optimizer
- **Regularization:** Dropout layers and L2 regularization to prevent overfitting.
- **Metrics:** Accuracy, F1 Score, Typing Speed (LPM), ITR.
- **Training:** The model is trained for 20 epochs with a batch size of 32.

## Results
- **Accuracy:** ~92.5%
- **F1 Score:** ~0.91
- **Typing Speed:** 10.94 letters/min
