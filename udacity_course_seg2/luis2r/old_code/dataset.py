import re
import cv2
from glob import glob
import os.path
import tensorflow as tf
import numpy as np
import scipy.misc
data_folder = "/home/shared/datasets/kitti_road/data_road/training"
import imageio
from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean
filenames = glob(os.path.join(data_folder, 'image_2', '*.png'))
#print (filenames)
filenames = tf.constant(filenames)
labels =    glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))
labels = tf.constant(labels)
print (filenames)
#labels = { re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path  for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
print (labels)


image_shape = (160, 576)


# Use a custom OpenCV function to read the image, instead of the standard
# TensorFlow `tf.read_file()` operation.
#def _read_py_function(filename, label):
#  image_decoded = cv2.imread(filename.decode(), cv2.IMREAD_GRAYSCALE)
#  label_decoded = cv2.imread(label.decode(), cv2.IMREAD_GRAYSCALE)
#  return image_decoded, label_decoded

# Use standard TensorFlow operations to resize the image to a fixed shape.
#def _resize_function(image_decoded, label_decoded):
#  image_decoded.set_shape([None, None, None])
#  image_resized = tf.image.resize_images(image_decoded, [28, 28])

#  label_decoded.set_shape([None, None, None])
#  label_resized = tf.image.resize_images(label_decoded, [28, 28])
#  return image_resized, label_resized


# Use a custom OpenCV function to read the image, instead of the standard
# TensorFlow `tf.read_file()` operation.
def _read_py_function(filename, label):
    #image_decoded = cv2.imread(filename.decode(), cv2.IMREAD_GRAYSCALE)
    #image_decoded = cv2.imread(filename.decode())
    #print(filename.decode())
    image_decoded = io.imread(filename.decode())
    #print('hello')
    #print(image_decoded.shape )
    #print(image_decoded)
    image_resized = resize(image_decoded, image_shape)


    label_decoded = io.imread(label.decode())
    label_resized = resize(label_decoded, image_shape)
    #print (image_resized.shape)
    #return image_decoded, label
    gt_bg = np.all(label_resized == background_color, axis=2)
    gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
    gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
    #print('gt_bg')
    #print(gt_bg)
    #print('gt_image')
    #print(gt_image)
    print (image_resized.shape)
    print (gt_image.shape)
    return image_resized, gt_image
    #return image_decoded, image_decoded

    #return image_decoded, image_decoded

# Use standard TensorFlow operations to resize the image to a fixed shape.
# def _resize_function(image_decoded, label):
#     #image_decoded.set_shape(160, 576)
#     #image_resized = tf.image.resize_images(image_decoded, [28, 28])
#     #image_resized = tf.image.resize_images(image_decoded, image_shape)

#     #image_resized = resize(image_decoded, image_shape, anti_aliasing=True) skimage 0.14
#     #image_resized = resize(image_decoded, image_shape)
#     image_resized = resize(image_decoded, (160, 576))
#     # gt_bg = np.all(image_resized == background_color, axis=2)
#     # gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
#     # gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
#     # print('gt_bg')
#     # print(gt_bg)
#     # print('gt_image')
#     # print(gt_image)
#     #return image_resized, label
#     return image_decoded, image_decoded










background_color = np.array([255, 0, 0])

#def _parse_function(filename, label):
    #image_string = tf.read_file(filename)
    #image_decoded = tf.image.decode_png(image_string)
    #image_resized = tf.image.resize_images(image_decoded, [160, 576])
    #print(image_resized)
    #label_string = tf.read_file(label)
    #label_decoded = tf.image.decode_png(label_string)
    #label_resized = tf.image.resize_images(label_decoded, [160, 576])


    #image = scipy.misc.imresize(scipy.misc.imread(filename.decode()), image_shape)
    #gt_image = scipy.misc.imresize(scipy.misc.imread(label.decode()), image_shape)

    #gt_bg = np.all(gt_image == background_color, axis=2)
    #gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
    #gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)




    #gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
    #gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

    #images.append(image)
    #gt_images.append(gt_image)






    #return image, gt_image
#filenames = ["/home/shared/datasets/kitti_road/data_road/training/training.txt","/home/shared/datasets/kitti_road/data_road/training/gt_training.txt"]
#labels    = ["/home/shared/datasets/kitti_road/data_road/training/gt_training.txt"]
#dataset = tf.data.TextLineDataset(filenames)
#dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))



dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
print("func")
#dataset = dataset.map(lambda filename, label: tuple(tf.py_func( _read_py_function, [filename, label], [tf.uint8, label.dtype])))
dataset = dataset.map(lambda filename, label: tuple(tf.py_func( _read_py_function, [filename, label], [tf.double, tf.bool])))

#dataset = dataset.map(_resize_function)



#dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
#dataset = dataset.map(_parse_function)


batched_dataset = dataset.batch(4)

with tf.Session() as sess:
    iterator = batched_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    #print(sess.run(next_element))  # 
    a,b = sess.run(next_element)
    print("a",a[0]) 
    print("b",b[0]) 

#print(sess.run(next_element))  # 
    a,b = sess.run(next_element)
    print("A",a[0]) 
    print("B",b[0])

#print(sess.run(next_element))  # 
    a,b = sess.run(next_element)
    print("C",a[0]) 
    print("D",b[0])
    #print(sess.run(next_element))  
