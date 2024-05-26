# Object Detection Repository
Welcome to the Object Detection repository! This project implements state-of-the-art models for identifying and localizing objects in images. It supports various popular architectures and provides all necessary tools for training, evaluation, and deployment.

# Features
Multiple Architectures: Support for YOLO, SSD, and Faster R-CNN.
Pre-trained Models: Access to pre-trained weights for quick testing and fine-tuning.
Training Scripts: Comprehensive scripts to train models from scratch.
Evaluation Tools: Tools to evaluate model performance using standard metrics.
Documentation: Detailed documentation and examples to get you started.
# Table of Contents
* Installation
* Usage
# Installation :

## Clone the repository and install the required dependencies:

### bash
#### Copy code
  git clone : https://github.com/Nikmal8/OBJECT_DETECTION.git

    cd object_detection
    pip install -r requirements.txt

# Usage :

     To use a pre-trained model for object detection, run:

### bash
#### Copy code
    python detect.py --model [MODEL_NAME] --image [IMAGE_PATH]
#### Example:

### bash
#### Copy code
    python detect.py --model yolov3 --image data/sample.jpg
