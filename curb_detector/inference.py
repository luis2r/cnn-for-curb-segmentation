import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

from tensorflow.python.tools import inspect_checkpoint as chkp

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))











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
tests.test_layers(layers)













# logits = None
# keep_prob = None
# input_tensor = None

def run():
    # global session
    # global logits
    # global keep_prob
    # global input_tensor



    num_classes = 2
    image_shape = (160, 576)
    shape_org = (256,256)
    data_dir = '/home/shared/datasets/bird_eye_view/velodyne'


    runs_dir = './runs'
    #tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    # helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/
    # saver = tf.train.Saver()
    # logits = tf.reshape(nn_last_layer, (-1, num_classes))

    with tf.Session() as sess:
        # Path to vgg model
        # vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        # get_batches_fn = helper.gen_batch_function(os.path.join(data_dir), image_shape)
        #get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function

        # epochs = 50
        # epochs = 50
        # batch_size = 16

        # TF placeholders
        # correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        # learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        sess.run(tf.global_variables_initializer())
        new_saver = tf.train.import_meta_graph('./models/model_E0001-B0005.ckpt.meta')
        new_saver.restore(sess, './models/model_E0001-B0005.ckpt')

        graph = tf.get_default_graph()

        #logits = graph.get_tensor_by_name('logits:0')
        # logits = graph.get_collection("logits")[0]
        # keep_prob = graph.get_tensor_by_name('keep_prob:0')
        # input_image = graph.get_tensor_by_name('image_input:0')




        # vgg_tag = 'vgg16'
        vgg_input_tensor_name = 'image_input:0'
        vgg_keep_prob_tensor_name = 'keep_prob:0'
        vgg_layer3_out_tensor_name = 'layer3_out:0'
        vgg_layer4_out_tensor_name = 'layer4_out:0'
        vgg_layer7_out_tensor_name = 'layer7_out:0'

        # tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
        input_image = graph.get_tensor_by_name(vgg_input_tensor_name)
        keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
        vgg_layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
        vgg_layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
        vgg_layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
        
        # return image_input, keep_prob, layer3_out, layer4_out, layer7_out






        # input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)

        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

        logits = tf.reshape(nn_last_layer, (-1, num_classes))
        # logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function
        # saver = tf.train.Saver()

        # train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
        #      correct_label, keep_prob, learning_rate)

		#saver.save(sess, '../models/segmentation_model.ckpt')

        # chkp.print_tensors_in_checkpoint_file("/tmp/model.ckpt", tensor_name='', all_tensors=True)



        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image, shape_org)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
