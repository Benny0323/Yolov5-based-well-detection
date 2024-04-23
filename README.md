# The hidden danger of manhole cover detection based on KDWC-YOLOv5(Knowledge Distillation Well Cover-YOLOv5)
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
Due to the fact that our proposed model comprises two stages, you need to download both stages' checkpoints to successfully run the codes!
These two files can be found in the following link : 

https://drive.google.com/file/d/1fclRgDYc_duWns63MbTeKRffmSPdP7BA/view?usp=sharing

### Evaluation
To do the evaluation process, first run the following command in stage 1 (the conditional diffusion model):
```
python Test.py
```      
Then, you will get a series of images generated by the conditional diffusion model. After that, run the following command in stage 2 with these images as inputs.
```
python Hybrid_autoencodereval.py
```
## Training by yourself
Our model‘s best checkpoint is located at the link below, you can download it freely.

