import os.path
import tensorflow as tf
import helper
import warnings
import scipy.misc
import numpy as np
from moviepy.editor import VideoFileClip
from distutils.version import LooseVersion
import project_tests as tests


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
    vgg_tag = 'vgg16'
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()

    vgg_input_tensor_name       = 'image_input:0'
    vgg_keep_prob_tensor_name   = 'keep_prob:0'
    vgg_layer3_out_tensor_name  = 'layer3_out:0'
    vgg_layer4_out_tensor_name  = 'layer4_out:0'
    vgg_layer7_out_tensor_name  = 'layer7_out:0'

    input_tensor      = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob_tensor  = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out_tensor = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out_tensor = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out_tensor = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return input_tensor, keep_prob_tensor, layer3_out_tensor, layer4_out_tensor, layer7_out_tensor


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    l2_reg = 1e-5

    # vgg_layer7_out = tf.Print(vgg_layer7_out, [tf.shape(vgg_layer7_out)]);
    layer7_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, strides=(1, 1), padding='same',
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))


    # vgg_layer4_out = tf.Print(vgg_layer4_out, [tf.shape(vgg_layer4_out)]);
    layer4_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, strides=(1, 1), padding='same',
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))

    # vgg_layer3_out = tf.Print(vgg_layer3_out, [tf.shape(vgg_layer3_out)]);
    layer3_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, strides=(1, 1), padding='same',
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))


    output = tf.layers.conv2d_transpose(layer7_1x1, num_classes, 4, strides=(2, 2), padding='same',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
    # output = tf.Print(output, [tf.shape(output)])

    output = tf.add(output, layer4_1x1)
    output = tf.layers.conv2d_transpose(output, num_classes, 4, strides=(2, 2), padding='same',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
    # output = tf.Print(output, [tf.shape(output)])

    output = tf.add(output, layer3_1x1)
    output = tf.layers.conv2d_transpose(output, num_classes, 16, strides=(8, 8), padding='same',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
    # output = tf.Print(output, [tf.shape(output)])

    return output


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name='logits')
    correct_label = tf.reshape(correct_label, (-1, num_classes))

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=correct_label, logits=logits))
    l2_loss = tf.losses.get_regularization_loss()

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss + l2_loss)

    return logits, train_op, cross_entropy_loss


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, saver):
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

    for i in range(epochs):
        for image, label in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={input_image: image, correct_label: label, keep_prob: 0.5, learning_rate:1e-4})

        print('Epoch: {}, Loss: {}'.format(i, loss))

    saver.save(sess, './kitti_saves/fcn.ckpt')


def train():
    batch_size = 4
    epochs = 20
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    learning_rate = tf.placeholder(dtype=tf.float32)
    correct_label = tf.placeholder(dtype=tf.int32, shape=(None, None, None, num_classes))

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        input_tensor, keep_prob_tensor, layer3_out_tensor, layer4_out_tensor, layer7_out_tensor = load_vgg(sess, vgg_path)
        output = layers(layer3_out_tensor, layer4_out_tensor, layer7_out_tensor, num_classes)
        logits, train_op, cross_entropy_loss = optimize(output, correct_label, learning_rate, num_classes)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_tensor, correct_label, keep_prob_tensor, learning_rate, saver)

        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob_tensor, input_tensor)
        # OPTIONAL: Apply the trained model to a video


def run_inference_test():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'

    learning_rate = tf.placeholder(dtype=tf.float32)
    correct_label = tf.placeholder(dtype=tf.int32, shape=(None, None, None, num_classes))

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        sess.run(tf.global_variables_initializer())
        new_saver = tf.train.import_meta_graph('./kitti_saves/fcn.ckpt.meta')
        new_saver.restore(sess, './kitti_saves/fcn.ckpt')

        graph = tf.get_default_graph()

        logits = graph.get_tensor_by_name('logits:0')
        keep_prob_tensor = graph.get_tensor_by_name('keep_prob:0')
        input_tensor = graph.get_tensor_by_name('image_input:0')

        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob_tensor, input_tensor)


def test():
    tests.test_load_vgg(load_vgg, tf)
    tests.test_layers(layers)
    tests.test_optimize(optimize)
    tests.test_train_nn(train_nn)


# Awful - can't pass parameters to fl_image, use global
session = None
logits = None
keep_prob = None
input_tensor = None
image_shape = (160, 576)


def inference_pipeline(image):
    global session
    global logits
    global keep_prob
    global input_tensor
    global image_shape

    image = scipy.misc.imresize(image, image_shape)

    im_softmax = session.run(
        [tf.nn.softmax(logits)],
        {keep_prob: 1.0, input_tensor: [image]}
    )
    im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
    segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    street_im = scipy.misc.toimage(image)
    street_im.paste(mask, box=None, mask=mask)

    output_image = np.array(street_im)
    return output_image


def run_video_test():
    global session
    global logits
    global keep_prob
    global input_tensor

    project_output = './kitti_video.mp4'

    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    clip = VideoFileClip("./project_video.mp4") #.subclip(0,1)
    # clip = VideoFileClip("./project_video.mp4")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        new_saver = tf.train.import_meta_graph('./kitti_saves/fcn.ckpt.meta')
        new_saver.restore(sess, './kitti_saves/fcn.ckpt')

        graph = tf.get_default_graph()

        session = sess
        logits = graph.get_tensor_by_name('logits:0')
        keep_prob = graph.get_tensor_by_name('keep_prob:0')
        input_tensor = graph.get_tensor_by_name('image_input:0')

        output_clip = clip.fl_image(inference_pipeline) #NOTE: this function expects color images!!

        output_clip.write_videofile(project_output, audio=False)


if __name__ == '__main__':
    run_video_test()
    # test()
    # run_inference_test()
    # train()

