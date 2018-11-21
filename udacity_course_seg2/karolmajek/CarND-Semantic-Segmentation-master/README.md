# Semantic Segmentation
## Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

![Segmentation](images/1504860116.0556026.gif)

Figure: Results after 10 epochs of training with augmentation (Command to reproduce: `python main.py 10 16 0.5`)

Check road segmentation with 4k video on youtube:

[![Road segmentation for Self-Driving Car - Dataset #2](http://img.youtube.com/vi/35DnxJl0aqA/0.jpg)](http://www.youtube.com/watch?v=35DnxJl0aqA)


## Setup
### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)

### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

#### 4K video data

```
wget https://drive.google.com/open?id=0B_6iW8KaJFXOQmhaWU56dlBDY28
```

## Training

Run the following command to run the project:

```
python main.py
```

Parameters:

- number of epoch
- batch size
- keep probability

```
#python main.py  EPOCHS  BATCH_SIZE  DROPOUT
python main.py 10 16 0.5
```

![Training loss](images/chart.png)

Figure: Training loss after 10 epochs of training with augmentation (Command to reproduce: `python main.py 10 16 0.5`)

## Data augmentation

`get_batches_fn()` in `helper.py`

- flipping every image
- translating +/- 100 in x axis and +/- 30 in y axis
- brightness +/- 150 (in 0-255 range)

Example batch:

![Example batch](images/batch.jpg)
