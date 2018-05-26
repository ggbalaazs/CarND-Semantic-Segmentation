# Semantic Segmentation

[//]: # (Image References)
[image1]: ./loss_chart.png "Training Loss"
[image2]: ./runs/1527249070.3695695/um_000005.png
[image3]: ./runs/1527249070.3695695/um_000010.png
[image4]: ./runs/1527249070.3695695/um_000012.png
[image5]: ./runs/1527249070.3695695/umm_000089.png
[image6]: ./runs/1527249070.3695695/uu_000050.png

### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

### Neural Network
Pretrained VGG model is loaded into TensorFlow - retrieving `input`, `keep`, `layer3`, `layer4` and `layer7` tensors. The fully connected layer is replaced by one-by-one convolutional layer. Then three transposed convolutional layers are added with skip connection layers in-between from `layer4` and `layer3`.
In the training phase pixel-wise cross entropy loss is calculated, L2 weight normalization is used and SGD is performed by `AdamOptimizer`. Actual loss is logged and can be tracked for current batches/epochs.

These are the chosen hyperparameters:
```Python
p = {
    'num_classes': 2,
    'img_shape': (160, 576),
    'batch_size': 5,
    'epochs': 60,
    'learning_rate': 0.00005,
    'keep_prob': 0.75
}
```

On average, the model decreases loss over time:
![alt text][image1]


Some inference samples of the final model (`runs/1527249070.3695695/`):
![image2]
![image3]
![image4]
![image5]
![image6]

### Setup
##### GPU
`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.
