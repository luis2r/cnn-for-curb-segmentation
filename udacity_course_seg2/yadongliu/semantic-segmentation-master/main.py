import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import numpy as np

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

    input_tensor = tf.get_default_graph().get_tensor_by_name(vgg_input_tensor_name)
    prob_tensor = tf.get_default_graph().get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 = tf.get_default_graph().get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = tf.get_default_graph().get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = tf.get_default_graph().get_tensor_by_name(vgg_layer7_out_tensor_name)

    return input_tensor, prob_tensor, layer3, layer4, layer7
#tests.test_load_vgg(load_vgg, tf)

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

    layer7_up = tf.layers.conv2d_transpose(vgg_layer7_out, 512, (2, 2), (2, 2),
        kernel_initializer= tf.random_normal_initializer(stddev=0.0001),
        kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3), name="layer7_2x")

    layer4_sum = tf.add(vgg_layer4_out, layer7_up)
    layer4_up = tf.layers.conv2d_transpose(layer4_sum, 256, (2, 2), (2, 2),
        kernel_initializer= tf.random_normal_initializer(stddev=0.0001),
        kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3), name="layer4_2x")

    layer3_sum = tf.add(vgg_layer3_out, layer4_up)
    layer3_up = tf.layers.conv2d_transpose(layer3_sum, num_classes, (8, 8), (8, 8),
        kernel_initializer= tf.random_normal_initializer(stddev=0.0001),
        kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3), name="layer3_8x")

    return layer3_up 
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
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    ## GradientDescentOptimizer v.s. AdamOptimizer
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss
## tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
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
    sess.run(tf.global_variables_initializer())

    for i in range(epochs):
        print('Epoch: ', (i+1))
        step = 0
        for images, gt_images in get_batches_fn(batch_size):
            step = step + 1
            _, cross_loss = sess.run([train_op, cross_entropy_loss],
                                     feed_dict={correct_label: gt_images,
                input_image: images, keep_prob: 0.5, learning_rate: 0.001})
            print("Loss at step ", step, " :", cross_loss)

#tests.test_train_nn(train_nn)


def run(inference = True):
    num_classes = 2
    image_shape = (160, 576)
    batch_size = 8
    epochs = 50
    data_dir = './data'
    runs_dir = './runs'
    # tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    learning_rate = tf.placeholder(tf.float32, name="learning_rate")
    # keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    correct_label = tf.placeholder(tf.float32, shape=(None, 160, 576, 2), name="label_image")
    # input_image   = tf.placeholder(tf.float32, shape=(None, 160, 576, 2), name="input_image")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_tensor, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
        print("input_tensor.get_shape(): ", input_tensor.get_shape())
        print("layer3.get_shape(): ", layer3.get_shape())
        print("layer4.get_shape(): ", layer4.get_shape())
        print("layer7.get_shape(): ", layer7.get_shape())

        fcnlayer_out = layers(layer3, layer4, layer7, num_classes)
        print("fcnlayer_out.get_shape(): ", fcnlayer_out.get_shape())

        logits, train_op, cross_entropy_loss = optimize(fcnlayer_out, correct_label, learning_rate, num_classes)
        print("logits.get_shape(): ", logits.get_shape())

        print(tf.trainable_variables())

        """
        Figuring out the shape of each layers
        """
        sess.run(tf.global_variables_initializer())
        test_images = ['data/data_road/testing/image_2/um_000000.png', 'data/data_road/testing/image_2/um_000000.png']
        input_images = helper.gen_input_tensor(test_images, image_shape)
        l3, l4, l7, fcn_out = sess.run([layer3, layer4, layer7, fcnlayer_out], feed_dict={input_tensor: input_images, keep_prob: 0.5})
        print('layer3 Shape should be (?, 20, 72, 256), actual: ', np.shape(l3))
        print('layer4 Shape should be (?, 10, 36, 512), actual: ', np.shape(l4))
        print('layer7 Shape should be (?, 5, 18, 4096), actual: ', np.shape(l7))
        print('fcn_out Shape should be (?, 160, 576, 2), actual: ', np.shape(fcn_out))

        # TODO: Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_tensor, correct_label, keep_prob, learning_rate)

        if inference: 
            saver = tf.train.Saver()
            save_path = saver.save(sess, "model.ckpt")
            print("Model saved in file: %s" % save_path)
            print("Generating inference samples ... ")
            helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_tensor)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run(True)
