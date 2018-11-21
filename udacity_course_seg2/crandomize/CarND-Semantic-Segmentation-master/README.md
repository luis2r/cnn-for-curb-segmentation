# Semantic Segmentation
### Introduction
In this project, we identify the pixels of a road in images using a Fully Convolutional Network (FCN) based on the VGG16 image classifier architecture. The following paper was used as a reference architecture used in this project ([FCN for Semantic Segmentation](https://arxiv.org/pdf/1605.06211.pdf)), which basically consists in a encoder using a pretrained VGG model and a decoder part which provides 1x1 convolutions and upsampling.

In order to train the data we use the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php)  

The following parameters were used:
 - Learning rate: 1e-4
 - Keep Probability: 70%
 - Epochs: 20
 - Batch size: 5

 ### Output

 The following images represent the pixels (colored in green) found as road by the FCN.

[//]: # (Image References)


[image1]: ./examples/um_000000.png "Image"
[image2]: ./examples/um_000003.png "Image"
[image3]: ./examples/um_000061.png "Image"
[image4]: ./examples/umm_000037.png "Image"
[image5]: ./examples/uu_000024.png "Image"

![Example 1][image1]
![Example 2][image2]
![Example 3][image3]
![Example 4][image4]
![Example 5][image5]


### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 

