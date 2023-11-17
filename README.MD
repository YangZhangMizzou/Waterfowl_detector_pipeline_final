# Waterfowl Detector Pipeline Instructions

This code serves as a demo showcasing the application of waterfowl detectors (RetinaNet, YOLOv5, FasterRCNN, and YoloNAS) and a waterfowl classifier (Res18 and MixMatch) on the Waterfowl dataset.

## System requirements
Support is available only for the Linux Ubuntu system, and it can be used within a CUDA environment. It has been tested in Ubuntu 18 with Python 3.8.

## Example images

This demo includes various images for testing. The example_images folder contains different images for each corresponding pre-trained model, organized by their folder names (Bird_A, Bird_B, etc.), along with the attached Ground Truth (GT) file for later evaluation.

## Installation

### Clone the repository
You can use the command line to clone the repository:
```
git clone https://github.com/YangZhangMizzou/Waterfowl_detector_pipeline_final.git
```

### Create virtual environment
Using a virtual environment is recommended for installing the package. Anaconda is recommended for this purpose. After installing Anaconda, refer to this guide to create your virtual environment. It's recommended to create the environment with Python 3.8:

```
conda create -n torch_py3 python==3.8
conda activate torch_py3
cd Waterfowl_detector_pipeline_final
```

### Install pytorch

We recommend installing PyTorch with CUDA to accelerate running speed:
```
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```
### Install basic dependency

```
pip install pandas
pip install numpy
pip install opencv-python
pip install tqdm
pip install efficientnet_pytorch
pip install resnet_pytorch
pip install scikit-learn
```

### Install dependency for FasterRCNN

```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### Install dependency for YOLOV5

Just run this command and all set for YOLOV5
```
pip install -r requirements.txt
```

### Install dependency for Retinanet

```
pip install opencv-contrib-python
pip install Pillow==9.5
pip install pyexiv2
pip install matplotlib
pip install -i https://test.pypi.org/simple/ WaterFowlTools
pip install packaging
pip install kiwisolver
pip install cycler
```


### Install dependency for YoloNAS and MixMatch

```
pip install super-gradients==3.1.3
pip install setuptools==59.5.0
pip install tensorboardX
pip install progress
```

## input format
This script requires a speical format of input describes below
```
Image_folder (eg Bird_A)
├── image_name1.jpg
├── image_name1.txt # if want to evaluate
├── image_name1.txt # if want to evaluate on classificaiton
└── ...
```
## Run all experiments:
You can easily run all experiments on the drone_collection dataset (all heights, habitats, datasets) with a single command. Please follow the procedures carefully:

1. Download [example_images.zip](https://drive.google.com/file/d/1nwzvAKL_fBVeFviAeDub3tpSmx-p_3ME/view?usp=share_link) and then extract all subfolders to **example_images** folder. 
2. Download [checkpoints.zip](https://drive.google.com/file/d/1gCochdduiTb7sxrAkGTR-DS_YZTEmbLi/view?usp=drive_link) and then extract all subfolders to **checkpoint** folder 
3. change directory to Waterfowl_detector_pipeline and run your customized commands(The inference may take a long time. Please be patient).

## Run the Scripts:
Once you have the input file ready and the correct virtual environment set up, you can use the file inference_image_height.py to start inferring the images. Here's a quick example of the full command:
```
python inference_image_height.py \
--det_model retinanet \
--image_root ./example_images/Bird_G_UAV_TestTrain \
--image_ext JPG \
--out_dir ./result/retinanet/Bird_G_UAV_TestTrain \
--evaluate True \
```

if you want to run waterfowl dectection and classification together:
python inference_image_height.py \
--det_model retinanet \
--cla_model mixmatch \
--image_root ./example_images/Bird_G_UAV_TestTrain \
--image_ext JPG \
--out_dir ./result/retinanet_mixmatch/Bird_G_UAV_TestTrain \
--evaluate True \
```


The description of each command are as follows:
```
--det_model: name of the detection model. You can select from yolo5, fasterrcnn, retinanetknn, retinanet, megadetector, and yolonas.
--cla_model: name of the classification model. You can select from res18 and mixmatch.
--image_root: specify where the input images are stored.
--image_ext: image extension of the target images, default is 'JPG'.
--image_date: specify the date the image was taken; this will be stored as description data.
--image_location: where the image is taken; this will be stored as description data.
--csv_root: The root dir where image info is stored.
--out_dir: where the output file will be generated. By default, it will create a 'Result' folder under the current directory.
--evaluate: whether we want to evaluate the result. This can only be done when the input file comes with a groundTruth file; default is false.
```
## Output format
When you specify the output dir, you shall expecting the output in the following:
```
Result folder 
├── detection-results
│   ├── image_name1.txt
│   ├── image_name2.txt
│   ├── image_name3.txt
│   └── ...
├── visualize-results
│   ├── image_name1.jpg
│   ├── image_name2.jpg
│   ├── image_name3.jpg
│   └── ...
├── configs.json
├── detection_summary.csv
├── f1_score.jpg    #if apply evaluation
└── mAP.jpg         #if apply evaluation

detection_summary contains three types of data:
Description data includes input info of the image such as image_name, date, altitude.
Metadata includes metadata read from the image metadata (if applicable).
Sample results are shown below:
Detection data: includes the number of birds detected and time spent inferring that image (including visualization).

Each txt file under detection_results file contains detected bounding boxes in the following format:
  category, confidence score, x1, y1, x2, y2
Sorted in confidence descending order.
```
When the user chooses to detect waterfowl only, the visualization of the result should look like this. Rectangles represent TP (True Positive), triangles represent FP (False Positive), and ovals represent FN (False Negative).
![I_DJI_0261](https://github.com/YangZhangMizzou/Waterfowl_detector_pipeline_final/assets/47132186/65b4ada5-ddce-483b-814e-e7c0d4473ffe)

When the user chooses to detect and classify waterfowl, the visualization of the result should look like this. Blue rectangles represent TP with correct class prediction, while green represent TP with incorrect class prediction
![DJI_0055](https://github.com/YangZhangMizzou/Waterfowl_detector_pipeline_final/assets/47132186/c5d5778b-369b-4071-a3e3-a6c41ff36cdb)

We generate confusion matrix to illustrate accuracy across different classes
![confusion_matrix](https://github.com/YangZhangMizzou/Waterfowl_detector_pipeline_final/assets/47132186/c68eab59-9c74-4a9a-b641-736cee1aed25)





