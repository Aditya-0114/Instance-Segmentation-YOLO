# Instance Segmentation with YOLOv8

This repository contains an instance segmentation project using the YOLOv8m-seg model. The dataset for this project is fetched from [PixelLib](https://github.com/ayoolaolafenwa/PixelLib), and the model is trained and evaluated using the YOLOv8 framework. The trained model is also exported to ONNX format for deployment purposes.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Training](#training)
- [Prediction](#prediction)
- [Exporting Model](#exporting-model)
- [Results](#results)
## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/instance-segmentation-yolov8.git
    cd instance-segmentation-yolov8
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Dataset

The dataset is fetched from the [PixelLib](https://github.com/ayoolaolafenwa/PixelLib) repository. Follow these steps to download and prepare the dataset:

1. Download the dataset from the [PixelLib](https://github.com/ayoolaolafenwa/PixelLib) repository.
2. Place the dataset in the `data/` directory.
3. Ensure the dataset structure matches the format expected by YOLOv8.

## Training

To train the YOLOv8 model for instance segmentation, use the following command:

```sh
yolo task=segment mode=train epochs=100 data=dataset.yaml model=yolov8m-seg.pt imgsz=640 batch=8
```

--task=segment: Specifies the task type as segmentation.
--mode=train: Specifies the mode as training.
--epochs=100: Number of training epochs.
--data=dataset.yaml: Path to the dataset configuration file.
--model=yolov8m-seg.pt: Pre-trained YOLOv8m-seg model.
--imgsz=640: Image size.
--batch=8: Batch size.

## Prediction
To make predictions using the trained model, run the predict.py script:

```sh
python predict.py --weights best.pt --source data/images
```

--weights best.pt: Path to the trained model weights.
--source data/images: Path to the directory containing images for prediction.

## Exporting Model
To export the trained model to ONNX format, use the following command:

```sh
yolo export model=best.pt format=onnx
```

--model=best.pt: Path to the trained model weights.
--format=onnx: Specifies the export format as ONNX.

## Results
The results of the training and evaluation will be saved in the runs/ directory. You can visualize the results using tools like TensorBoard.
