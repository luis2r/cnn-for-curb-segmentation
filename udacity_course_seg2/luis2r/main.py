import multiprocessing
import re
import os.path
import tensorflow as tf
import helper2
import warnings
from distutils.version import LooseVersion
import project_tests as tests
from glob import glob
import numpy as np
from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean
import cv2
import scipy.misc
from PIL import Image
# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    image_input = tf.get_default_graph().get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = tf.get_default_graph().get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = tf.get_default_graph().get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = tf.get_default_graph().get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = tf.get_default_graph().get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    # 1x1 convolution of vgg layer 7
    layer7a_out = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, 
                                   padding= 'same', 
                                   kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                   kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    # upsample
    layer4a_in1 = tf.layers.conv2d_transpose(layer7a_out, num_classes, 4, 
                                             strides= (2, 2), 
                                             padding= 'same', 
                                             kernel_initializer= tf.random_normal_initializer(stddev=0.01), 
                                             kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    # make sure the shapes are the same!
    # 1x1 convolution of vgg layer 4
    layer4a_in2 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, 
                                   padding= 'same', 
                                   kernel_initializer= tf.random_normal_initializer(stddev=0.01), 
                                   kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    # skip connection (element-wise addition)
    layer4a_out = tf.add(layer4a_in1, layer4a_in2)
    # upsample
    layer3a_in1 = tf.layers.conv2d_transpose(layer4a_out, num_classes, 4,  
                                             strides= (2, 2), 
                                             padding= 'same', 
                                             kernel_initializer= tf.random_normal_initializer(stddev=0.01), 
                                             kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    # 1x1 convolution of vgg layer 3
    layer3a_in2 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, 
                                   padding= 'same', 
                                   kernel_initializer= tf.random_normal_initializer(stddev=0.01), 
                                   kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    # skip connection (element-wise addition)
    layer3a_out = tf.add(layer3a_in1, layer3a_in2)
    # upsample
    nn_last_layer = tf.layers.conv2d_transpose(layer3a_out, num_classes, 16,  
                                               strides= (8, 8), 
                                               padding= 'same', 
                                               kernel_initializer= tf.random_normal_initializer(stddev=0.01), 
                                               kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    return nn_last_layer
#tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    # make logits a 2D tensor where each row represents a pixel and each column a class
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1,num_classes))
    # define loss function

    #cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= logits, labels= correct_label))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits= logits, labels= correct_label))
    # define training operation
    optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate, iterator,seed):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    next_element = iterator.get_next()
    sess.run(tf.global_variables_initializer())
    
    print("Training...")
    #print()
    for i in range(epochs):
        #sess.run(iterator.initializer)
        print("EPOCH {} ...".format(i+1))
        #for image, label in get_batches_fn(batch_size):
        #for i in batch_size:
        #    _, loss = sess.run([train_op, cross_entropy_loss], 
        #                       feed_dict={input_image: image, correct_label: label, keep_prob: 0.5, learning_rate: 0.0009})
        #    print("Loss: = {:.3f}".format(loss))
        #images = []
        #gt_images = []
        # Compute for 100 epochs.

        #sess.run(iterator.initializer)
        sess.run(iterator.initializer, feed_dict={seed: i})

        while True:
            try:
                image,label = sess.run(next_element)

                #print("build batch")

                #print('image',image)
                #print('shape',image[0].shape)
                #print('label',label[0].shape)
                #images.append(image)
                #gt_images.append(label[0])
                #print(len(image))
                #print(len(label))

            except tf.errors.OutOfRangeError:
                break
        #images_np = np.array(images)
        #gt_images_np = np.array(gt_images)
        #print('shape images',images_np.shape)
        #print('shape images',images_np.shape)
            #print('loss')
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict={input_image: image, correct_label: label, keep_prob: 0.5, learning_rate: 0.0009})
        ##########
            print("Loss: = {:.5f}".format(loss))





#tests.test_train_nn(train_nn)

# background_color = np.array([255, 0, 0])

# def _parse_function(filename, label):
#     print('file',filename)
#     print('label',label)
# image_shape = (160, 576)
    
#     image_string = tf.read_file(filename)
#     image_decoded = tf.image.decode_png(image_string)
#     #image_resized = tf.image.resize_images(image_decoded, [4, 4])
#     image_resized = tf.image.resize_images(image_decoded, image_shape)

#     label_string = tf.read_file(label)
#     label_decoded = tf.image.decode_png(label_string)
#     label_resized = tf.image.resize_images(label_decoded, image_shape)

#     print('label', label_resized)
#     print('background_color',background_color)
#     gt_bg = np.all(label_resized == background_color, axis=2)
#     gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
#     gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
#     #images.append(image)
#     #gt_images.append(gt_image)
#     return image_resized, gt_image

# def _read_resize_py_function(filename, label):
#     background_color = np.array([255, 0, 0])
#     image_shape = (160, 576)
#     # image_shape = (40, 128)
#     #image_decoded = cv2.imread(filename.decode(), cv2.IMREAD_GRAYSCALE)
#     #image_decoded = cv2.imread(filename.decode())
#     #print(filename.decode())
#     image_decoded = io.imread(filename.decode())
#     #print('hello')
#     #print(image_decoded.shape )
#     #print(image_decoded)
#     image_resized = resize(image_decoded, image_shape)

#     label_decoded = io.imread(label.decode())
#     label_resized = resize(label_decoded, image_shape)
#     #print (image_resized.shape)
#     #return image_decoded, label
#     gt_bg = np.all(label_resized == background_color, axis=2)
#     gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
#     gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
#     #print('gt_bg')
#     #print(gt_bg)
#     #print('gt_image')
#     #print(gt_image)
#     #print (image_resized.shape)
#     #print (gt_image.shape)
#     return image_resized, gt_image



# def _read_resize_py_function(filename):

#     background_color = np.array([255, 0, 0])

#     image_shape = (160, 576)

#     folder_img = "/home/shared/datasets/kitti_road/data_road/training/image_2"
#     image_decoded = io.imread(folder_img+"/"+filename.decode())
#     print(folder_img+"/"+filename.decode())
#     image_resized = resize(image_decoded, image_shape)
#     #print("dataset", folder_img+"/"+filename.decode())
#     filename_gt = re.sub(r'(?is)_', '_road_', filename.decode())
#     folder_gt = "/home/shared/datasets/kitti_road/data_road/training/gt_image_2" 
#     label_decoded  = io.imread(folder_gt+"/"+filename_gt)
#     print(folder_gt+"/"+filename_gt)
#     label_resized = resize(label_decoded, image_shape)
#     #print (image_resized.shape)
#     #return image_decoded, label
#     gt_bg = np.all(label_resized == background_color, axis=2)
#     gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
#     gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
#     #print('gt_bg')
#     #print(gt_bg)
#     #print('gt_image')
#     #print(gt_image)
#     #print (image_resized.shape)
#     #print (gt_image.shape)
#     #return image_resized, gt_image
#     return image_resized, gt_image

def _read_resize_py_function(filename):

    #background_color = np.array([0, 0, 255])
    background_color = np.array([255, 0, 0])#scipy.misc

    image_shape = (576,160)
    folder_img = "/home/shared/datasets/kitti_road/data_road/training/image_2"

    #image_decoded = scipy.misc.imread(folder_img+"/"+filename.decode())
    #image_decoded = cv2.imread(folder_img+"/"+filename.decode())
    #image_decoded = Image.open(folder_img+"/"+filename.decode())
    image_decoded = Image.open(folder_img+"/"+filename.decode())
    #print(folder_img+"/"+filename.decode())
    #image_resized = scipy.misc.imresize(image_decoded, image_shape)
    #image_resized = cv2.resize(image_decoded, image_shape)
    #image_resized = Image.resize(image_decoded, image_shape)
    image_decoded=image_decoded.resize(image_shape)
    image_resized = np.array(image_decoded)
    #print("dataset", folder_img+"/"+filename.decode())
    filename_gt = re.sub(r'(?is)_', '_road_', filename.decode())
    folder_gt = "/home/shared/datasets/kitti_road/data_road/training/gt_image_2" 
    #label_decoded  = scipy.misc.imread(folder_gt+"/"+filename_gt)
    #label_decoded  = cv2.imread(folder_gt+"/"+filename_gt)
    label_decoded  = Image.open(folder_gt+"/"+filename_gt)
    
    #print(folder_gt+"/"+filename_gt)
    #label_resized = scipy.misc.imresize(label_decoded, image_shape)
    #label_resized = cv2.resize(label_decoded, image_shape)
    #label_resized = Image.resize(label_decoded, image_shape)
    label_decoded =label_decoded.resize( image_shape)
    label_resized = np.array(label_decoded)
    #print(label_resized.shape)
    #print("show")
    #label_decoded.show()
    #print (label_resized)
    gt_bg = np.all(label_resized == background_color, axis=2)
    #print(gt_bg.shape)
    #print(gt_bg[0][0])
    gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
    #print(gt_bg[0][0])
    gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
    gt_image = np.concatenate((gt_image, np.invert(gt_image)), axis=2)
    gt_image = np.concatenate((gt_image, np.invert(gt_image)), axis=2)
    gt_image = np.concatenate((gt_image, np.invert(gt_image)), axis=2)
    gt_image = np.concatenate((gt_image, np.invert(gt_image)), axis=2)
    gt_image = np.concatenate((gt_image, np.invert(gt_image)), axis=2)
    gt_image = np.concatenate((gt_image, np.invert(gt_image)), axis=2)
    #print(gt_image[100][100])
    #print(gt_image.shape)
    #print("max",np.max(label_resized))
    #print("min",np.min(label_resized))
    #print("gt max",np.max(gt_image))
    #print("gt min",np.max(gt_image))
    #return folder_img+"/"+filename.decode(), folder_gt+"/"+filename_gt
    #print("shape image",image_decoded.shape)
    #print("shape gt image",gt_image.shape)
    print("gt shape",gt_image.shape)
    print("gt",gt_image)
    print("ax2",np.size(gt_image,axis=2))
    print("ax1",np.size(gt_image,axis=1))
    print("ax0",np.size(gt_image,axis=0))
    return image_decoded, gt_image



# def input_pipeline(filenames, batch_size, num_shards, seed=None):
#     #dataset = tf.data.Dataset.list_files(filenames).shuffle(num_shards)
#     #dataset = dataset.interleave( lambda filename: (tf.data.TextLineDataset(filename) .skip(1) .map(lambda row: parse_csv(row, hparams), num_parallel_calls=multiprocessing.cpu_count())), cycle_length=5) 
#     #dataset = dataset.interleave( lambda filename: (tf.data.TextLineDataset(filename).map(lambda filename: tuple(tf.py_func( _read_resize_py_function, [filename], [tf.double, tf.bool])), num_parallel_calls=multiprocessing.cpu_count())), cycle_length=5) 
#     dataset = (tf.data.TextLineDataset(filenames).map(lambda filename: tuple(tf.py_func( _read_resize_py_function, [filename], [tf.double, tf.bool])), num_parallel_calls=multiprocessing.cpu_count()))#, cycle_length=5) 
#     dataset = dataset.shuffle(buffer_size=10000, seed=seed)
#     #dataset = tf.data.TextLineDataset(filenames)
#     #dataset = dataset.map(lambda filename: tuple(tf.py_func( _read_resize_py_function, [filename], [tf.double, tf.bool])))
#     batched_dataset = dataset.batch(batch_size)
#     #dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
#     #dataset = dataset.map(lambda filename, label: tuple(tf.py_func( _read_resize_py_function, [filename, label], [tf.double, tf.bool])))
#     #dataset = dataset.shuffle(buffer_size=10000)
#     #batched_dataset = dataset.batch(batch_size)
#     #iterator = batched_dataset.make_one_shot_iterator()
#     #iterator = batched_dataset.make_initializable_iterator()
#     return batched_dataset.make_initializable_iterator()

def input_pipeline(filenames, batch_size, num_shards, seed=None):

    dataset = tf.data.Dataset.list_files(filenames).shuffle(num_shards)
    #dataset = dataset.interleave( lambda filename: (tf.data.TextLineDataset(filename) .skip(1) .map(lambda row: parse_csv(row, hparams), num_parallel_calls=multiprocessing.cpu_count())), cycle_length=5) 
    #dataset = dataset.interleave( lambda filename: (tf.data.TextLineDataset(filename).map(lambda filename: tuple(tf.py_func( _read_resize_py_function, [filename], [tf.double, tf.bool])), num_parallel_calls=multiprocessing.cpu_count())), cycle_length=5) 
    dataset = dataset.interleave( lambda filename: (tf.data.TextLineDataset(filename).map(lambda filename: tuple(tf.py_func( _read_resize_py_function, [filename], [tf.uint8, tf.bool])), 
        num_parallel_calls=multiprocessing.cpu_count())), cycle_length=5) 
    #dataset = dataset.interleave( lambda filename: (tf.data.TextLineDataset(filename).map(lambda filename: tuple(tf.py_func( _read_resize_py_function, [filename], [tf.string, tf.string])), num_parallel_calls=2)), cycle_length=2) 
    dataset = dataset.shuffle(buffer_size=1000, seed=seed)
    batched_dataset = dataset.batch(batch_size)

    return batched_dataset.make_initializable_iterator()






def run():
    num_classes = 128
    image_shape = (576,160)
    data_dir = '/home/shared/datasets/kitti_road'
    runs_dir = './runs'
    filenames = ["/home/shared/datasets/kitti_road/data_road/training/training.txt"]
    filenames_gt = ["/home/shared/datasets/kitti_road/data_road/training/gt_training.txt"]
    # data_folder = "/home/shared/datasets/kitti_road/data_road/training"
    # filenames = glob(os.path.join(data_folder, 'image_2', '*.png'))
    # filenames = tf.constant(filenames)
    # labels =    glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))
    # labels = tf.constant(labels)
    # print(labels)

    ###########################################################   random.shuffle(filenames)

    #tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper2.maybe_download_pretrained_vgg(data_dir)

    epochs = 50
    batch_size = 4
    num_shards = 1
    #seed = None




    #dataset = tf.data.Dataset.list_files(filenames).shuffle(num_shards)


    #dataset = dataset.interleave( lambda filename: (tf.data.TextLineDataset(filename) .skip(1) .map(lambda row: parse_csv(row, hparams), num_parallel_calls=multiprocessing.cpu_count())), cycle_length=5) 

    #dataset = dataset.interleave( lambda filename: (tf.data.TextLineDataset(filename) .skip(1) .map(lambda filename: tuple(tf.py_func( _read_resize_py_function, [filename], [tf.double, tf.bool])), num_parallel_calls=multiprocessing.cpu_count())), cycle_length=5) 


    #dataset = dataset.shuffle(buffer_size=10000, seed=20)


    #dataset = tf.data.TextLineDataset(filenames)
    #dataset = dataset.map(lambda filename: tuple(tf.py_func( _read_resize_py_function, [filename], [tf.double, tf.bool])))

    #batched_dataset = dataset.batch(batch_size)




    #dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    #dataset = dataset.map(lambda filename, label: tuple(tf.py_func( _read_resize_py_function, [filename, label], [tf.double, tf.bool])))
    #dataset = dataset.shuffle(buffer_size=10000)
    #batched_dataset = dataset.batch(batch_size)
    #iterator = batched_dataset.make_one_shot_iterator()
    #iterator = batched_dataset.make_initializable_iterator()
    
    seed = tf.placeholder(tf.int64, shape=())
    iterator = input_pipeline(filenames, batch_size, num_shards, seed)

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches

        #get_batches_fn = helper2.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)



        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function




        # TF placeholders
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)

        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function

        #train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate)
        train_nn(sess, epochs, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate, iterator, seed)

        # TODO: Save inference data using helper2.save_inference_samples
        helper2.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
