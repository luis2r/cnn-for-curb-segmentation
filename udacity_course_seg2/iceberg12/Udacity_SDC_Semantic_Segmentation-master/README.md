# Semantic Segmentation
[architecture1]: ./images/architecture1.png
[bike]: ./images/bike.png
[VGGfreeze]: ./images/VGGfreeze.png
[um_000003_0]: ./images/um_000003_0.png
[um_000003_1]: ./images/um_000003_1.png
[um_000003_2]: ./images/um_000003_2.png


### Introduction
In this project, I'll label the pixels of a road in images using a [Fully Convolutional Network (FCN)](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf). The components of a FCN includes a pretrained neural network (ie. VGG-16) as an encoder and transpose convolution layers as a decoder. The role of the encoder is capturing the features present at different depths (layer 3, 4, 7), while the decoder adds the upsampled (transposed) final layer 7 with the skip connections produced by putting layers 3 and 4 through 1x1 convolutions. Helpful animations of convolutional operations, including transposed convolutions, can be found [here](https://github.com/vdumoulin/conv_arithmetic). The final sum has the same size of the original image and it is used to predict whether each pixel belongs to a labeled class.

Simplified FCN Structure
![alt text][architecture1]

With Skip Connections
![alt text][bike]

Note: There are two options for transfer learning in semantic segmentation, depending on whether the pretrained network parameters are going to be re-trained by the new data. If we want to fix the pretrained network, we need to freeze the backpropagation at all skip connections (layer 3, 4 and 7 of VGG).

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Goal

The project labels most pixels of roads close to the best solution. The model doesn't have to predict correctly all the images, just most of them. A solution that is close to best would label at least 80% of the road and label no more than 20% of non-road pixels as road.

### Implementation

#### Training

The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip). This model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. I extract them by name.

```python
tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)
graph = tf.get_default_graph()
    
# get the layers of vgg to use in Fully Convolutional Network FCN-8s
input_ = graph.get_tensor_by_name('image_input:0')
keep_prob = graph.get_tensor_by_name('keep_prob:0')
layer3 = graph.get_tensor_by_name('layer3_out:0')
layer4 = graph.get_tensor_by_name('layer4_out:0')
layer7 = graph.get_tensor_by_name('layer7_out:0')
```

The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 

```python
layer_3 = tf.multiply(layer_3, 0.0001)
layer_4 = tf.multiply(layer_4, 0.01)
```    

Now come the interesting part of convolution layer 1x1. I want to set L2 regularizer for the weights in each of these layers, with the weights are initialized in an educated manner (truncated_normal, with std 0.01 to avoid large values which cause exploding later). We have

```python
layer_4 = tf.layers.conv2d(vgg_layer7_out, num_classes, kernel_size=1, padding='same',
    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
```

and add it to the **transposed** (or upsampled) of layer 7.

```python
decoding_layer_1 = tf.layers.conv2d_transpose(layer_7, num_classes, 
    kernel_size=4, strides=2, padding='same', 
    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
decoding_layer_1 = tf.add(decoding_layer_1, layer_4)
```

Note, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.

```python
regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
regularization_loss = sum(regularization_losses)
...
cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))    
cross_entropy_loss = tf.add(cross_entropy_loss, regularization_loss)
train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)
```

Yeah, you are good to go with that! Just one more thing, try to limit the batch_size in 5 to 10 images to avoid Out-of-memory error for your graphic card (mine is the humble GTX 1060).

#### Testing

Note that fixing the pretrained network can be done by stopping the gradient flows.

```python
vgg_layer7_out = tf.stop_gradient(vgg_layer7_out)
vgg_layer4_out = tf.stop_gradient(vgg_layer4_out)
vgg_layer3_out = tf.stop_gradient(vgg_layer3_out)
```

In addition, here is the part I find it very common but useful: how to save and load the Tensorflow model for scoring purpose. After struggling a while, I figure out the template.

```python
## Train and save the model
tf.reset_default_graph()
sess.run(tf.global_variables_initializer())
... # create the model, naming all operations you want to extract
... # logits = tf.reshape(nn_last_layer, (-1, num_classes), name='logits')
...
saver = tf.train.Saver()
...  # train the model
saver.save(sess, 'segmentation_model')

## Load and score the model
# load
saver = tf.train.import_meta_graph('segmentation_model.meta')
saver.restore(sess,tf.train.latest_checkpoint('./'))
# extract tensors, including input tensor and hyperparameter tensor.
graph = tf.get_default_graph()
input_image = graph.get_tensor_by_name('image_input:0')
keep_prob = graph.get_tensor_by_name('keep_prob:0')
logits = graph.get_tensor_by_name('logits:0')  # note that here we call the TENSOR as the result of operation, not the operation itself. Call it by operation_name::x.
... # load a sample image
# score
sess.run([tf.nn.softmax(logits)], {keep_prob: 1.0, input_image: [image]})
```

In the Jupyter notebook, there is also a code for scoring through frames of a video, overlay the classification and output the video result.

### Dead or Alive?

What I found interesting with the limited time is the difference in fixing the pretrained VGG or not. It is not a  question. Here is the findings

- Fixing the pretrained VGG really boosts up the training time, reducing from 33s per epoch to 10s per epoch.
- However, the training error comes down slower. This is explanable since the network is too frigid to tune, but it might become better will more epochs.

![alt text][VGGfreeze]

Both the networks can classify the images at the rate 4.5 frame/s, which is ok but not at 24 frame/s though :P. Here is the classification result from the unfixed FCN, step-by-step during fine-tuning.

Original

![alt text][um_000003_0]

After l2 regularization

![alt text][um_000003_1]

After skip connection

![alt text][um_000003_2]

### Next

If having more time I will try to implement on other class detection (pedestrian, bicycle, sign board, ...). Also try to archive the test result at every epoch to improve the prediction accuracy in test.
