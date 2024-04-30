# The hidden danger of manhole cover detection based on KDWC-YOLOv5(Knowledge Distillation Well Cover-YOLOv5)
![image](https://github.com/Benny0323/Yolov5-based-well-detection/blob/main/demo.jpg)
### 🧨 Congratulations! We achieve 0.948 of mAP on 350 test images in a competition!
 To-do list:
 - [x] Model Ensembling
 https://github.com/ultralytics/yolov5/issues/318
 - [x] Shape IOU
 https://github.com/malagoutou/Shape-IoU
 - [x] Multi-scale training
 - [x] Knowledge distillation
 https://blog.roboflow.com/what-is-knowledge-distillation/
 - [x] kepp training model, delving into best epoch and hyperparameter settings
 - [ ] Publish dataets after the competition ends.
 - [x] Publish a web app made by my team.


## Pre-requisties
* Linux

* Python>=3.7

* NVIDIA GPU (memory>=30G) + CUDA cuDNN

## Strat evaluating
### Install dependencies
```
pip install -r requirements.txt
```
### Download the checkpoint
Our model‘s best checkpoint is located at the link below, you can download it freely.

https://drive.google.com/file/d/1fclRgDYc_duWns63MbTeKRffmSPdP7BA/view?usp=sharing

### Evaluation
If you want to get the mAP value, run the following command:
```
python val.py
```
If you want to get the images with bounding boxes, run the following command:
```
python detect.py
```
If you have a GPU cluster, I also provide you with a script file using sbatch to submit, and you can run it with:
```
sbatch yolov5_val.sh/sbatch yolov5_detect.sh
```
**Tips**: You can also use "--" to add parameters in the running command according to yourslef.
E.g. If I want to output a txt file with the order of "Image name Confidence coefficient Coordinates", you can run the command below:
```
python detect.py --save-txt --save-conf
```
## Training by yourself
If you want to train our model by yourself, you should firstly change the specify the **path** of your dataset in "data/A30.yaml", 
and you also nedd to specify a pretrained model, we use [yolov5m](https://drive.google.com/file/d/16h2MhkAz4ntuPk4sySABDakP8O8uSw4m/view?usp=sharing), or you can choose other pretrained model via [official link](https://github.com/ultralytics/yolov5), then can run the following command:
```
python train.py
```
Tip1: You can also use "--" to add multi-scale parameters in the running command if you want to multi-scale training:
```
python train.py --multi-scale
```
Tip2: It is better to put the pretrained model under the root directory.
