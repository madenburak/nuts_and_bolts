# YOLOv8 + ByteTrack

## Create Python Virtual Environment

    I suggest create new python virtual environment and work in it's to you. Installiations can disorder your python packages.

    For create:
    `python -m venv new_env`

    For activate:
    `.\new_env\Scripts\activate`

## Data Preparing

    There are labels as coco json format on our hands. For YOLO training, we must convert it to yolo format. We can use `dataset_manager.py` for this operation. The python file provides some convenience to us.

    * Convert COCO to YOLO format
    * Counting the number of labels for each class
    * Masking object that at chosen class on images and deleting labels in .txt files
    * Convert source of video to images


## Detection and Tracking

    You can follow the `detect_and_track.ipynb` file for training for detection and adding track feature on structure.


![](.\images\result.gif) 