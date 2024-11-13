# Waterfowl Detection and Classification Pipeline Instructions

This code implements local waterfowl detectors (RetinaNet, YOLOv5, FasterRCNN, and YoloNAS) and waterfowl classifiers (Res18 and MixMatch) which are pretrained by waterfowl datasets collected by Missouri Department of Conservation and University of Missouri,Columbia.

## System requirements
Support is available for the Linux Ubuntu system and Windows system, It has been tested in Ubuntu 18, 20.4 and windows 10 and 11.

## Example images

All Example images can be downloaded from [example_images.zip](https://drive.google.com/file/d/1GpPj6GQl_-oaCb7y-YwId4sUjyLDvipQ/view?usp=sharing). Download and  extract all subfolders to **example_images** folder. 

## Installation

### 1.Install CUDA

This software supports CUDA to accelerate inference speed. You can follow the instructions [here](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) to customize your CUDA installation. This software has been tested and confirmed to work with all version of cuda 10 and 11. If you choose not to install CUDA, this software can work with cpu at a slow inference speed.

### 2.Clone the repository

Git must be installed first to download the code locally. You can follow the instructions [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) to install Git on your operating system. Once installed, you can use the command line to clone the repository

You can use the command line to clone the repository:
```
git clone https://github.com/YangZhangMizzou/Waterfowl_detector_pipeline_final.git
```

### 3.Install Anaconda and Create virtual environment

Anaconda is used to create virtual environment for this software. Guideline for installation on windows and Ubuntu can be found [here](https://docs.anaconda.com/anaconda/install/linux/). After installing Anaconda, refer to this guide to create your virtual environment. It is recommended to create the environment using Python 3.8, as this is the version used during the software’s development.

```
conda create -n torch_py3 python==3.8
conda activate torch_py3
cd Waterfowl_detector_pipeline_final
```

### 4.Install pytorch
We recommend installing PyTorch with CUDA to accelerate running speed by using GPU resource:
```
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```
You can also install PyTorch without CUDA. In this case, the software will run with cpu resource:
```
pip install torch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0
```

### 5.Install basic dependency

```
pip install pandas
pip install numpy
pip install opencv-python
pip install tqdm
pip install efficientnet_pytorch
pip install resnet_pytorch
pip install scikit-learn
```

### 6.Install dependency for FasterRCNN

```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### 7.Install dependency for YOLOV5

```
pip install -r requirements.txt
```

### 8.Install dependency for Retinanet

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


### 9.Install dependency for YoloNAS and MixMatch

```
pip install super-gradients==3.1.3
pip install setuptools==59.5.0
pip install tensorboardX
pip install progress
```

## Run the Scripts
Once you have prepared the input file and finish installing all dependency, you can use **inference_image_height.py**  to start the image inference process.

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
When you specify the **output_dir** when running **inference_image_height.py**, you shall expecting the output in the following:
```
Output folder 
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
└── detection_summary.csv 
```

**detection_summary.csv** contain image information and image inference result summary of all images. **Image Information** includes details such as image_name, image_altitude, and other metadata extracted from the images. **Inference Results** provides the number of detected birds and the inference time for each image
![199340134-13dc5f02-4980-4bac-9a6a-4a5d6a04050e](https://github.com/user-attachments/assets/3f9bd18c-58aa-4617-abe5-e077f5ff1910)

Each **image_name.txt** file in the detection-results folder contains the coordinates of all predicted bounding boxes (bboxes) for that image. Each bbox is represented by five values: inference confidence, x and y coordinates of the bbox's top-left point, and the bbox's width and height. If you choose to run the waterfowl classifier, an additional value indicating the bird category will be included for each bbox.

Each **image_name.jpg** file in the visualize-results provides a visualization of all predicted bounding boxes. Predicted birds detected by the waterfowl detector are marked with green bounding boxes. If you choose to run the classifier, the waterfowl species name will be displayed near each green bounding box. The image below shows an example.
![20241113004900](https://github.com/user-attachments/assets/b033de5f-43c1-47ad-8c0e-ff39d0ac85cd)








