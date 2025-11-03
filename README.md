# Zero-Shot Object Detection with Grounding DINO

This repository contains a Jupyter notebook (DINO.ipynb) that explores DINO (self-supervised Vision Transformer) for attention map visualization and Grounding DINO for zero-shot object detection. It demonstrates how self-supervised models learn semantic representations and how grounding models detect objects based on free-form text prompts without task-specific training. The notebook includes code for loading models, processing images/text, performing detection, and visualizing results.

The notebook is designed for educational purposes, providing insights into self-supervised learning and multimodal grounding in PyTorch and Hugging Face Transformers.

## Table of Contents

*   Project Overview
*   Features
*   Requirements
*   Installation
*   Usage
*   Results and Analysis
*   Contributing
*   License

## Project Overview

*   **Part 1: Visualizing Attention Maps in DINO**
    *   Overview of DINO's self-distillation architecture (student-teacher setup).
    *   Load pre-trained DINO model and preprocess images.
    *   Extract and visualize attention maps from the \[CLS\] token to understand model focus on semantic regions.
    *   Discuss why DINO captures object-level semantics without labels.
*   **Part 2: Zero-Shot Object Detection with Grounding DINO**
    *   Introduction to Grounding DINO: Combines DINO with grounded pre-training for text-based object detection.
    *   Setup: Clone Grounding DINO repo and load model/tokenizer.
    *   Preprocess images and text prompts (e.g., "a person with a hat").
    *   Perform detection to get bounding boxes, logits, and phrases.
    *   Post-process results (NMS, thresholding) and visualize with bounding boxes and labels.
    *   Examples: Detect objects in various images using custom prompts.
    *   Analysis: Strengths (flexibility, zero-shot) and limitations (accuracy on complex scenes).

No external datasets are required; the notebook uses sample images from URLs or local paths.

## Features

*   Attention map extraction and visualization for self-supervised insights.
*   Zero-shot detection: Detect arbitrary objects via text prompts without fine-tuning.
*   Customizable thresholds for box and text confidence.
*   Visualization utilities: Draw bounding boxes, labels, and confidence scores.
*   End-to-end examples with real-world images.
*   Discussion questions on model behavior, limitations, and applications.

## Requirements

*   Python 3.x
*   PyTorch
*   Torchvision
*   Transformers (Hugging Face)
*   Matplotlib
*   OpenCV (cv2)
*   NumPy
*   Supervision (for annotations)

See the notebook's import and installation sections for details:
```python
import torch
from torch import nn
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import cv2
import supervision as sv
```

## Installation

1.  Clone the repository:
    ```sh
    git clone https://github.com/MohammadJavadShamloo/Dino.git
    cd Dino
    ```
    
2.  Install dependencies:
    ```sh
    pip install torch torchvision transformers matplotlib numpy opencv-python supervision
    ```
    
3.  Clone and install Grounding DINO (as per notebook):
    ```sh
    git clone https://github.com/IDEA-Research/GroundingDINO.git
    cd GroundingDINO
    pip install -q -e .
    ```
    
4.  (Optional) Use a virtual environment:
    ```sh
    python -m venv env
    source env/bin/activate  # On Linux/Mac
    .\env\Scripts\activate   # On Windows
    pip install -r requirements.txt  # Create this file with the above libraries
    ```
    

## Usage

1.  Open the Jupyter notebook:
    ```sh
    jupyter notebook DINO.ipynb
    ```
    
2.  Run the cells sequentially:
    *   Part 1: Load DINO, process images, extract/visualize attention maps.
    *   Part 2: Set up Grounding DINO, load model, preprocess inputs.
    *   Perform detection on sample images with custom prompts.
    *   Visualize results and experiment with thresholds/prompts.

Note: GPU is recommended for faster inference (set device = 'cuda' if available). Download pre-trained weights as instructed in the notebook.

## Results and Analysis

*   **Attention Maps**: Visualizations show DINO focusing on objects/parts, revealing emergent semantic understanding.
*   **Zero-Shot Detection**: Examples detect objects like "red car" or "person with hat" with bounding boxes; confidence scores indicate reliability.
*   **Performance**: High flexibility for arbitrary prompts; limitations include potential false positives in complex scenes or low-confidence detections.
*   **Insights**: Discusses self-supervised learning benefits, grounding advantages, and real-world applications (e.g., robotics, search).
*   Refer to the notebook for plots, detected images, and question-based analysis.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for improvements, bug fixes, or additional examples like new prompts or datasets.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
