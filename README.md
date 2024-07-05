# Instance Segmentation with YOLOv8

This repository contains code and instructions for performing instance segmentation on a dataset fetched from [PixelLib](https://github.com/ayoolaolafenwa/PixelLib). The model is trained using YOLOv8's `yolov8m-seg` model. The repository also includes code for making predictions and exporting the trained model to ONNX format.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Training](#training)
- [Prediction](#prediction)
- [Exporting Model](#exporting-model)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
## Overview

Instance segmentation is a task in computer vision that involves detecting objects in an image and segmenting them into individual instances. This project utilizes the YOLOv8 architecture to perform instance segmentation on a custom dataset.

## Dataset

The dataset used for this project is fetched from the [PixelLib repository](https://github.com/ayoolaolafenwa/PixelLib). Follow the instructions in the PixelLib repository to download and prepare the dataset.

## Training

To train the instance segmentation model, the following command is used:

```bash
yolo task=segment mode=train epochs=100 data=dataset.yaml model=yolov8m-seg.pt imgsz=640 batch=8
This command trains the yolov8m-seg model for 100 epochs with an image size of 640 and a batch size of 8. The configuration for the dataset should be specified in the dataset.yaml file.

## Prediction
Predictions are made using the predict.py script. This script takes an input image and outputs the segmented instances.

## Exporting Model
After training, the model can be exported to the ONNX format using the following command:

```bash
yolo export model=yolov8m-seg.pt format=onnx

## Requirements
Python 3.7 or higher
YOLOv8
PyTorch
ONNX

## Installation
Clone the repository:

```bash
git clone https://github.com/your-username/instance-segmentation-yolov8.git
cd instance-segmentation-yolov8

 ## Install the required packages:

```bash
pip install -r requirements.txt

## Usage
Prepare the dataset as per the instructions in the PixelLib repository.

Train the model using the training command:

```bash
yolo task=segment mode=train epochs=100 data=dataset.yaml model=yolov8m-seg.pt imgsz=640 batch=8


## Make predictions using the predict.py script:

```bash
python predict.py --input path/to/image.jpg --output path/to/output.jpg

## Export the trained model to ONNX format:

```bash
yolo export model=yolov8m-seg.pt format=onnx

## Contributing
Contributions are welcome! Please fork this repository and submit a pull request for any enhancements or bug fixes.
