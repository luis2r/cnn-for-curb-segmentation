import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import scipy    


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
    print("Loading VGG")
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()

    input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer_3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer_4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer_7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    print("VGG loaded")

    return input, keep, layer_3, layer_4, layer_7


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    print("Creating layers")
    # Note: num_classes is binar: road vs. no-road
    # 1by1 convolution
    conv_1by1 = tf.layers.conv2d(vgg_layer7_out, num_classes, kernel_size=1, padding='same',
                                 kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # First transposed convolution for upscaling
    conv_trans_1 = tf.layers.conv2d_transpose(conv_1by1, num_classes, kernel_size=4, strides=(2, 2), padding='same',
                                              kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # 1by1 convolution vgg_layer4_out
    conv_1by1_vgg4 = tf.layers.conv2d(vgg_layer4_out, num_classes, kernel_size=1, padding= 'same',
                                      kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # First skip layer and further transposed convolution for upscaling
    skip_1 = tf.add(conv_trans_1, conv_1by1_vgg4)
    conv_trans_2 = tf.layers.conv2d_transpose(skip_1, num_classes, kernel_size=4, strides=(2, 2), padding='same',
                                              kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # 1by1 convolution vgg_layer3_out
    conv_1by1_vgg3 = tf.layers.conv2d(vgg_layer3_out, num_classes, kernel_size=1, padding= 'same',
                                      kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # Second skip layer and further transposed convolution for upscaling
    skip_2 = tf.add(conv_trans_2, conv_1by1_vgg3)
    output = tf.layers.conv2d_transpose(skip_2, num_classes, kernel_size=16, strides=(8, 8), padding='same',
                                        kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    print("Layers created")

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
    print("Setup optimizer")
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1,num_classes))

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)
    print("Optimizer setup")

    return logits, train_op, cross_entropy_loss


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate):
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
    print("Training NN")
    sess.run(tf.global_variables_initializer())

    epoch_loss = 0
    for epoch in range(epochs):
        batch = 0
        for image, label in get_batches_fn(batch_size):
            batch += 1
            _, loss = sess.run([train_op, cross_entropy_loss],
                                feed_dict = {input_image: image,
                                             correct_label: label,
                                             keep_prob: 0.5,
                                             learning_rate: 1e-3})

            epoch_loss += loss
            print("Batch {} - Current loss: {:.3f}".format(batch, loss))
        epoch_loss /= batch
        print("Epoch {} - Current loss: {:.3f}".format(epoch, epoch_loss))
    print("Training done")


def run():
    print("Run")
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
    # Note: Not done
    
    kBatchSize = 5
    kEpochs = 10

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
        # Note: Not implemented.

        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        # Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(layer_output, correct_label, learning_rate, num_classes)
        
        # Train NN using the train_nn function
        train_nn(sess, kEpochs, kBatchSize, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate)

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        
        # Save the variables to disk.
        print("Saving model...")
        saver = tf.train.Saver()
        save_path = saver.save(sess, "./model/semantic_segmentation_model.ckpt")
        print("Model saved in path: %s" % save_path)

        # OPTIONAL: Apply the trained model to a video
        # Note: For this, run the run_video method from main.

        
# Global sessio object for video processing
g_session = None
g_logits = None
g_keep_prob = None
g_input_image = None
 
def process_video_image(image, frame_name=""):
    image_shape = (160, 576)
    processed_image = helper.get_inference_samples_video(image, g_session, image_shape, g_logits, g_keep_prob, g_input_image)
    
    image_shape_orig = (360, 640)
    small = scipy.misc.imresize(processed_image, image_shape_orig)
    return small


def run_video():
    print("Run Video")

    from moviepy.editor import VideoFileClip

    file = "videos/challenge_video"
    
    clip = VideoFileClip("./" + file + ".mp4")
    output_video = "./" + file + "_processed.mp4"
    
    data_dir = './data'
    num_classes = 2

    global g_session
    global g_logits
    global g_keep_prob
    global g_input_image
    
    with tf.Session() as g_session:
        vgg_path = os.path.join(data_dir, 'vgg')

        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        g_input_image, g_keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(g_session, vgg_path)
        layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes)
        g_logits, train_op, cross_entropy_loss = optimize(layer_output, correct_label, learning_rate, num_classes)
        
        print("Restoring model...")
        saver = tf.train.Saver()
        saver.restore(g_session, "./model/semantic_segmentation_model.ckpt")
        print("Model restored.")

        output_clip = clip.fl_image(process_video_image)
        # output_clip = clip.subclip(0, 1).fl_image(process_video_image)
        output_clip.write_videofile(output_video, audio=False)
        

if __name__ == '__main__':
    tests.test_load_vgg(load_vgg, tf)
    tests.test_layers(layers)
    tests.test_optimize(optimize)
    tests.test_train_nn(train_nn)
    
    # Either call "run()" for training the CNN and saving the network or "run_video" to apply
    # the trained network onto a video.
    # For video: run_video()
    run()
