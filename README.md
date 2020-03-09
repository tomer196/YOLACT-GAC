# Segmentation Refinement with GAC

based on the code from [YOLACT: Real-time Instance Segmentation](https://arxiv.org/abs/1904.02689)

# Changes from the original YOLACT code
eval.py lines 411-426, adding the ability to do refinement with GAC  
train_GAC_loss.py lines 34-120, copy of train.py with the GAC loss, calaulate the distance maps and changing the data loader  
layers/modules/multibox_loss2.py lines 34-120, copy of multibox_loss.py with the added GAC loss  
few more minor changes to pass the distance maps to the loss module

# Installation
 - Clone this repository and enter it:
   ```Shell
   git clone https://github.com/tomer196/YOLACT-GAC.git
   cd YOLACT-GAC
   ```
 - Set up the environment using one of the following methods:
     - Set up a Python3 environment (e.g., using virtenv).
     - Install packages:
       ```Shell
       # Cython needs to be installed before pycocotools
       pip install Cython==0.29.15
       pip install -r requirments.txt 
       ```
 - download the COCO dataset and the 2014/2017 annotations. Note that this script will take a while and dump 21gb of files into `./data/coco`.
   ```Shell
   sh data/scripts/COCO.sh
   ```

# Training
 - To train, grab an imagenet-pretrained model and put it in `./weights`.
   - For Resnet50, download `resnet50-19c8e357.pth` from [here](https://drive.google.com/file/d/1Jy3yCdbatgXa5YYIdTCRrSV0S9V5g1rn/view?usp=sharing).
 - Run one of the training commands below.
   - All weights are saved in the `./weights` directory by default with the file name `<config>_<epoch>_<iter>.pth`.
```Shell
# Trains using the original losses
python train.py --config=yolact_resnet50_config

# Trains using the original losses + GAC loss
python train_GAC_loss.py --config=yolact_resnet50_config
```

# Evaluation
You can download the YOLACT trained models:

|               | Backbone |  Weights                        |
|:-------------:|:--------:|:--------------------------------|
| Original      | Resnet50 | [yolact_resnet50_54_800000.pth](https://drive.google.com/file/d/1yp7ZbbDwvMiFJEq4ptVKTYTI2VeRDXl0/view?usp=sharing) |
| With GAC loss | Resnet50 | [yolact_resnet50_GAC.pth](https://technionmail-my.sharepoint.com/:u:/g/personal/tomer-weiss_campus_technion_ac_il/EQ6o4OJjUq9CkevoLxlytJQBkketGNgaUOKm7Llvb70cjQ?e=dYIHPQ) | 

To evalute the model, put the corresponding weights file in the `./weights` directory and run one of the following commands. 

## Quantitative Results on COCO
```Shell
# Quantitatively evaluate the original trained model. 
python eval.py --trained_model=weights/yolact_resnet50_54_800000.pth

# Quantitatively evaluate my trained model with GAC loss. 
python eval.py --trained_model=weights/yolact_resnet50_GAC.pth
# or any train model in weights/

# to evaluate with the GAC refinement add the --GAC flag
python eval.py --trained_model=weights/yolact_resnet50_54_800000.pth --GAC
python eval.py --trained_model=weights/yolact_resnet50_GAC.pth --GAC
```
