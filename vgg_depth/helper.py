import re
import random
from glob import glob
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from urllib.request import urlretrieve
from tqdm import tqdm
import cv2
from PIL import Image
from scipy.sparse import csr_matrix, find


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))


# def gen_batch_function(data_folder, image_shape):
#    """
#    Generate function to create batches of training data
#    :param data_folder: Path to folder that contains all the datasets
#    :param image_shape: Tuple - Shape of image
#    :return:
#    """
#    def get_batches_fn(batch_size):
#        """
#        Create batches of training data
#        :param batch_size: Batch Size
#        :return: Batches of training data
#        """


#        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
#        label_paths = {
#            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
#            for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
#        background_color = np.array([255, 0, 0])

#        random.shuffle(image_paths)
#        for batch_i in range(0, len(image_paths), batch_size):
#            images = []
#            gt_images = []
#            for image_file in image_paths[batch_i:batch_i+batch_size]:
#                gt_image_file = label_paths[os.path.basename(image_file)]

#                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
#                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

#                gt_bg = np.all(gt_image == background_color, axis=2)
#                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
#                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

#                images.append(image)
#                gt_images.append(gt_image)

#            yield np.array(images), np.array(gt_images)
#    return get_batches_fn



def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    print("datafolder",data_folder)
    #for image_file in glob(os.path.join(data_folder, 'test/scene0712_00/color_rob/', '*.jpg')):
    for image_file in glob(os.path.join(data_folder, '*.jpg')):
        print(image_file)
        #image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
        #image = scipy.misc.imresize(scipy.misc.imread(image_file), (image_shape[1],image_shape[0]))#########funciona
        image = Image.open(image_file)
        #image= image.resize(image_shape)
        #image = np.array(image)
        image_shape = (576,160)
        new_size =(576,160)
        width_d, height_d = 640,480
        #width = width_d-576
        width = width_d-new_size[0]
        #height = height_d-160
        height = height_d-new_size[1]
        th_sky = 5 #threshold crop the sky
        #left = random.randint(5, width) 
        left = 0
        #top = random.randint(th_sky, height)
        top = 0
        right, bottom = left+new_size[0], top+new_size[1]





        cropped = image.crop( ( left, top, right, bottom ) )  # size: 576 X 160
        image_resized = np.array(cropped)
        #print("dataset", folder_img+"/"+filename.decode())

        print(np.max(image_resized),np.min(image_resized))

        a = [-128,-128,-128]
        image_norm = np.sum((image_resized,a),axis=0)

        image_norm = np.divide(image_norm,128)
        #cropped.save("nn.png")

        #print (image_norm)
        print(np.max(image_norm),np.min(image_norm))








        #image = cv2.imread(image_file)
        #image.resize(image_shape)
        #image = cv2.resize(image, image_shape)
        #print(logits.shape)
        #print(logits)
        #im_softmax = sess.run([tf.nn.softmax(logits)],{keep_prob: 1.0, image_pl: [image_norm]})#########funciona
        im_softmax = sess.run([logits],{keep_prob: 1.0, image_pl: [image_norm]})#########funciona
        print("softmax ",len(im_softmax[0]))
        print("softmax ",len(im_softmax[0][0]))
        print("softmax ",im_softmax)

        #print("softmax al ",im_softmax)
        
        im_softmax = np.reshape(im_softmax, (len(im_softmax[0]),len(im_softmax[0][0])),1)

        print("a",im_softmax.shape)
        #print("b",im_softmax)
        #im_argmax = np.argmax(im_softmax,axis=1)



        #print("c",im_argmax.shape)
        #print("d",im_argmax)
        #print(np.max(im_argmax))
        #im_argmax = np.reshape(im_argmax,(np.size(im_argmax)))
        #print("c1",im_argmax.shape)
        #print(im_argmax)
        #one_hot = np.eye(256)[im_argmax]
        #print(one_hot.shape)
        #one_hot_b=blockshaped(one_hot, 1242, 85)
        #one_hot_b=blockshaped(one_hot, image_shape[0], 85)

        #print(one_hot.shape)
        #one_hot_b=blockshaped(one_hot, image_shape[0], 85)
        #print(one_hot_b.dtype)
        #one_hot_b = one_hot.astype(bool)
        #print(one_hot_b.dtype)

        #print(one_hot_b.shape)
        #_,inverse_one_hot,_ = find(one_hot)
        #image_one_hot=im_softmax.reshape(image_shape[1],image_shape[0])
        
        #depth = im_softmax.reshape(160, 576,1)
        depth = np.multiply(im_softmax,128)
        a = [128]
        depth = np.sum((depth,a),axis=0)

        #depth = image_one_hot.astype(np.float) * 4



    #print("resta",np.max(image_norm),np.min(image_norm))


        depth = depth.reshape(160, 576)





        image_one_hot = depth
        im_out = np.uint8(image_one_hot)
        img = Image.fromarray(im_out, mode="P")
        #img.show()



        #im_softmax = im_softmax[0].reshape(image_shape[1], image_shape[0],128)#########funciona
        #print("softmax shape ",im_softmax.shape)
        
        

        #im_softmax = im_softmax[0][:, 1].reshape(image_shape[1], image_shape[0])



        #print("softmax col ",im_softmax.shape)#########funciona



        #segmentation = (im_softmax > 0.5).reshape(image_shape[1], image_shape[0], 1)#########funciona
        #print("segmantation ",segmentation)#########funciona
        #print("segmantation ",segmentation)
        #mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))#########funciona
        #print("shape ",mask.shape)
        #print("shape ",mask)


        #mask = scipy.misc.toimage(mask, mode="RGBA")#########funciona
        #mask = Image.fromarray(mask, mode="RGBA")

        #street_im = scipy.misc.toimage(image)#########funciona

        #street_im = Image.fromarray(image)
        #street_im.paste(mask, box=None, mask=mask)  #########funciona
        #street_im.paste(mask, box=None, mask=mask)  #########funciona
        #mask.astype(np.uint8)
        #print(np.array(street_im).shape)
        yield os.path.basename(image_file), np.array(img)#########funciona



def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(

        sess, logits, keep_prob, input_image, os.path.join(data_dir, 'test/scene0708_00/color_rob'), image_shape)
    for name, image in image_outputs:
        #print(image.shape)
        scipy.misc.imsave(os.path.join(output_dir, name), image)
        #image = Image.fromarray(image, mode="RGB")
        #image.save(os.path.join(output_dir, name))
        
        #cv2.imwrite(os.path.join(output_dir, name), image)
