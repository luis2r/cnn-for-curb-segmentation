# **Semantic Segmentation**
#### _Lorenzo's version_
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[sample1]: ./sample_1.png
[sample2]: ./sample_2.png


### Introduction
In this project, we label the pixels of a road in images using a Fully Convolutional Network (FCN).

#### Implementation details:
4 main functions are part of this project:
1. `load_vgg`: this function loads a pre-trained VGG network. VGG is well known for being capable of doing image classification and is commonly used for transfer learning.
2. `layers`: starting from the VGG network, we add, after the output layer, a 1x1 convolution followed by 2 steps of convolution transpose + skip layers. This function outputs the last layer of the Fully Convolutional network.
3.  `optimize`: this function defines the loss (cross-entropy loss) we are using in order to train our FCN, together with the optimization method (here: Adam Optimizer from TensorFlow)
4. `train_nn`: simply goes through our image batches and run the optimize node of our TensorFlow graph. It also prints the status of the training (current epoch + current loss)

#### Results
Below are 2 samples taken from the output of a network trained with 50 epochs, batch size 5, keep probability 0.5 and learning rate 0.001. Pixels marked in green are recognized as road:
![alt text][sample1]
![alt text][sample2]


##### Project Dependencies
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)

##### Dataset
We used the network to predict pixel images of the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  

##### Running the project
There are 2 ways to run the project:
- Semantic Segmentation notebook (messages are printed in the terminal)
- `python main.py`
