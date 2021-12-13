# EE541 Final Project

## Sartorius - Cell Instance Segmentation
### Detect single neuronal cells in microscopy images

```
Author: Jing Hu, Yankun Li, Ziyou Geng
```

## Requirements
```
matplotlib==3.4.3
numpy==1.21.0
opencv_python==4.5.4.60
pandas==1.3.4
Pillow==8.4.0
scikit_learn==1.0.1
tensorboardX==2.4.1
torch==1.8.0
torchvision==0.10.0
tqdm==4.62.3
```
  
## To train the model
```commandline
python train.py
```
```
options：
-e number of epochs,
-s start number of epochs,
-b batch size,
-l learning rate,
-c the path to load model,
-i init the model,
-f folder of train and validation images,
-k k-fold cross validation,
-n a integer parmeter between [0,k) of k-fold cross validation

```


## To predict from the model
```commandline
python inference.py
```
```
options：
-b batch size,
-c the path to load model,
-i init the model
-t folder of test images
-o the path of the submission file
```
The default path to store the training data images from sartorius is ```./data/sartorius/images```
