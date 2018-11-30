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


logits = None
keep_prob = None
input_tensor = None

def run():
    global session
    global logits
    global keep_prob
    global input_tensor



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

        # input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)

        # nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

        # logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function
        # saver = tf.train.Saver()

        # train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
        #      correct_label, keep_prob, learning_rate)

		#saver.save(sess, '../models/segmentation_model.ckpt')

        # chkp.print_tensors_in_checkpoint_file("/tmp/model.ckpt", tensor_name='', all_tensors=True)

        sess.run(tf.global_variables_initializer())
        #new_saver = tf.train.import_meta_graph('./models/model_E0001-B0005.ckpt.meta')
        new_saver.restore(sess, './models/model_E0001-B0005.ckpt')

        for i, var in enumerate(new_saver._var_list):
            print('Var {}: {}'.format(i, var))

        graph = tf.get_default_graph()

        # logits = graph.get_tensor_by_name('logits:0')
        keep_prob = graph.get_tensor_by_name('keep_prob:0')
        input_tensor = graph.get_tensor_by_name('image_input:0')

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image, shape_org)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
