#!/usr/bin/python

from PIL import Image
import numpy as np
#from sklearn.preprocessing import normalize
#from keras.utils import to_categorical

#from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csr_matrix, find

def blockshaped(arr, nrows, ncols):
    """
    https://stackoverflow.com/questions/16873441/form-a-big-2d-array-from-multiple-smaller-2d-arrays/16873755#16873755
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array looks like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def unblockshaped(arr, h, w):
    """
    https://stackoverflow.com/questions/16873441/form-a-big-2d-array-from-multiple-smaller-2d-arrays/16873755#16873755
    Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    n, nrows, ncols = arr.shape
    return (arr.reshape(h//nrows, -1, nrows, ncols)
               .swapaxes(1,2)
               .reshape(h, w))





def depth_read(filename, new_size):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt
    #depth_png = np.array(Image.open(filename), dtype=int)#####no rezize
    depth_png = np.array(Image.open(filename).resize(new_size), dtype=int)

    # make sure we have a proper 16bit depth map here.. not 8bit!
    #depth_png.show()
    assert(np.max(depth_png) > 255)
    depth = depth_png.astype(np.float) / 256.
    #depth[depth_png == 0] = -1.
    #depth = normalize(depth)
    depth = np.uint8(depth)

    return depth




filename = "/home/shared/datasets/depth_kitti/depth/depth_single_img/train/2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_03/0000000005.png"

new_size =(234,125)
image_arr = depth_read(filename, new_size)
img_in = Image.fromarray(image_arr, mode="P")
#img_in = img_in.resize((234,125))########################reshape
img_in.show()


###############one hot########################
img_array = np.reshape(image_arr,(np.size(image_arr)))
one_hot = np.eye(85)[img_array]
print(one_hot.shape)
#one_hot_b=blockshaped(one_hot, 1242, 85)
#one_hot_b=blockshaped(one_hot, 234, 85)
one_hot_b=blockshaped(one_hot, new_size[1], 85)


print(one_hot_b.dtype)
one_hot_b = one_hot_b.astype(bool)
print(one_hot_b.dtype)

print(one_hot_b.shape)

###############one hot########################



_,inverse_one_hot,_ = find(one_hot)
#i=i.reshape(375, 1242)
#image_one_hot=inverse_one_hot.reshape(125,234)
image_one_hot=inverse_one_hot.reshape(new_size[1],new_size[0])




im_out = np.uint8(image_one_hot)
img = Image.fromarray(im_out, mode="P")
img.show()
