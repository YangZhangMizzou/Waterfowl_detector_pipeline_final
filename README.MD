# Waterfowl Detection and Classification Pipeline Instructions

This code implements local waterfowl detectors (RetinaNet, YOLOv5, FasterRCNN, and YoloNAS) and waterfowl classifiers (Res18 and MixMatch) which are pretrained by waterfowl datasets collected by Missouri Department of Conservation and University of Missouri,Columbia.

## System requirements
Support is available for the Linux Ubuntu system and Windows system, It has been tested in Ubuntu 18, 20.4 and windows 10 and 11.

## Example images

All Example images can be downloaded from [example_images.zip](https://drive.google.com/file/d/1GpPj6GQl_-oaCb7y-YwId4sUjyLDvipQ/view?usp=sharing). Download and  extract all subfolders to **example_images** folder. 

## Installation

### Install CUDA

This software supports CUDA to accelerate inference speed. You can follow the instructions [here](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) to customize your CUDA installation. This software has been tested and confirmed to work with CUDA versions 11.3 and 10.1.

### Clone the repository

Git must be installed first to download the code locally. You can follow the instructions [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) to install Git on your operating system. Once installed, you can use the command line to clone the repository

You can use the command line to clone the repository:
```
git clone https://github.com/YangZhangMizzou/Waterfowl_detector_pipeline_final.git
```

### Install Anaconda and Create virtual environment

Anaconda is used to create virtual environment for this software. Guideline for installation on windows and Ubuntu can be found [here](https://docs.anaconda.com/anaconda/install/linux/). After installing Anaconda, refer to this guide to create your virtual environment. It's recommended to create the environment with Python 3.8:

```
conda create -n torch_py3 python==3.8
conda activate torch_py3
cd Waterfowl_detector_pipeline_final
```

### Install pytorch
We recommend installing PyTorch with CUDA to accelerate running speed by using GPU resource:
```
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```
You can also install PyTorch without CUDA. In this case, the software will run with cpu resource:
```
pip install torch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0
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

## Run the Scripts
Once you have prepared the input file and set up the correct virtual environment, you can use </ins>inference_image_height.py</ins>  to start the image inference process.

If you want run waterfowl detection on images only:
```
python inference_image_height.py \
--det_model retinanet \
--image_root ./example_images/Bird_A \
--image_ext JPG \
--out_dir ./result/retinanet/Bird_A \
```

if you want to run waterfowl dectection and classification on images:
```
python inference_image_height.py \
--det_model retinanet \
--cla_model mixmatch \
--image_root ./example_images/Bird_A \
--image_ext JPG \
--out_dir ./result/retinanet_mixmatch/Bird_A \
```

The description of each parameters are as follows:
```
--det_model: name of the detection model. You can select from yolo5, fasterrcnn, retinanetknn, retinanet, megadetector, and yolonas.
--cla_model: name of the classification model. You can select from res18 and mixmatch.
--image_root: specify where the input images are stored.
--image_ext: image extension of the target images, default is 'JPG'.
--image_date: specify the date the image was taken; this will be stored as description data.
--image_location: where the image is taken; this will be stored as description data.
--csv_root: The root dir where image info is stored.
--out_dir: where the output file will be generated. By default, it will create a 'Result' folder under the current directory.
```

## Inference output
When you specify the <ins>output_dir</ins> when running </ins>inference_image_height.py</ins>, you shall expecting the output in the following:
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
Description data includes input info of the image info such as image_name, date,altitude
Meta data includes meta data read from the image Meta data(if appliable)
Sample results are shown below:
Detection data: includes num of birds detected and time spent inferencing that image(include visualization)

When the user chooses to detect waterfowl, the visualization of the result should look like this. We use rectangle to mark each predictions
![DJI_0040](https://github.com/user-attachments/assets/0e6905a4-d9fd-4186-aceb-ad9eda5850d4)







