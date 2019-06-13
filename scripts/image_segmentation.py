#!/usr/bin/env python
# coding: utf-8

def get_image_data(image_id, image_type, rsz_ratio=1):
    """
    Method to get image data as np.array specifying image id and type
    """
    fname = get_filename(image_id, image_type)
    img = cv2.imread(fname)
    if rsz_ratio != 1:
        img = cv2.resize(img, dsize=(int(img.shape[1] * rsz_ratio), int(img.shape[0] * rsz_ratio)))
    assert img is not None, "Failed to read image : %s, %s" % (image_id, image_type)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# a channel saturation threshold
LOWER_A_SAT = 0
UPPER_A_SAT = 300

def Ra_space(img, Ra_ratio=1, a_upper_threshold=UPPER_A_SAT, a_lower_threshold=LOWER_A_SAT):
    imgLab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB);
    w = img.shape[0]
    h = img.shape[1]
    Ra = np.zeros((w*h, 2))
    for i in range(w):
        for j in range(h):
            R = math.sqrt((w/2-i)*(w/2-i) + (h/2-j)*(h/2-j))
            Ra[i*h+j, 0] = R
            a = min(imgLab[i][j][1], a_upper_threshold)
            a = max(imgLab[i][j][1], a_lower_threshold)
            Ra[i*h+j, 1] = a
            
    if Ra_ratio != 1:
        Ra[:,0] /= max(Ra[:,0])
        Ra[:,0] *= Ra_ratio
        Ra[:,1] /= max(Ra[:,1])

    return Ra


def crop_roi(image, display_image=False):
    """
    get the new images with ROI region showed
    
    """
    
    # creating the R-a feature for the image
    Ra_array = Ra_space(image)
    
    # k-means gaussian mixture model
    g = mixture.GaussianMixture(n_components = 2, covariance_type = 'diag', random_state = 0, init_params = 'kmeans')
    g.fit(Ra_array)
    labels = g.predict(Ra_array)
    
    # creating the mask array and assign the correct cluster label
    boolean_image_mask = np.array(labels).reshape(image.shape[0], image.shape[1])
    
    if display_image==True:
        outer_cluster_label = boolean_image_mask[0,0]
    
        new_image = image.copy()
    
        for i in range(boolean_image_mask.shape[0]):
            for j in range(boolean_image_mask.shape[1]):
                if boolean_image_mask[i, j] == outer_cluster_label:
                    new_image[i, j] = mask_color
    
#         plt.figure(figsize=(figure_size,figure_size))
    
#         plt.subplot(221)
#         plt.title("Original image")    
#         plt.imshow(image), plt.xticks([]), plt.yticks([])
    
#         plt.subplot(222)
#         plt.title("Region of interest")
#         plt.imshow(new_image), plt.xticks([]), plt.yticks([])
    
#         a_channel = np.reshape(Ra_array[:,1], (image.shape[0], image.shape[1]))
#         plt.subplot(223)
#         plt.title("a channel")
#         plt.imshow(a_channel, cmap='gist_heat'), plt.xticks([]), plt.yticks([])
  

    return new_image, boolean_image_mask


def get_filename(image_id, image_type):
    """
    Method to get image file path from its id and type   
    """
    if image_type == "Type_1" or         image_type == "Type_2" or         image_type == "Type_3":
        data_path = os.path.join(TRAIN_DATA, image_type)
    elif image_type == "Test":
        data_path = TEST_DATA
    elif image_type == "AType_1" or           image_type == "AType_2" or           image_type == "AType_3":
        data_path = os.path.join(ADDITIONAL_DATA, image_type[1:])
    else:
        raise Exception("Image type '%s' is not recognized" % image_type)

    ext = 'jpg'
    return os.path.join(data_path, "{}.{}".format(image_id, ext))


import os
from glob import glob
import cv2
import numpy as np
import math
import pandas as pd

from sklearn import mixture
import matplotlib.pylab as plt

from subprocess import check_output

TRAIN_DATA = " "
# check the types in training data
print(check_output(["ls", TRAIN_DATA]).decode("utf8"))

# get the original images
all_cervix_images = []

for path in sorted(glob(TRAIN_DATA + "*")):
    cervix_type = path.split("/")[-1]
    cervix_images = sorted(glob(TRAIN_DATA + cervix_type + "/*"))
    all_cervix_images = all_cervix_images + cervix_images

all_cervix_images = pd.DataFrame({'imagepath': all_cervix_images})
all_cervix_images['filetype'] = all_cervix_images.apply(lambda row: row.imagepath.split(".")[-1], axis=1)
all_cervix_images['file_id'] = all_cervix_images.apply(lambda row: row.imagepath.split("/")[-1].strip(".jpg"), axis=1)
all_cervix_images['type'] = all_cervix_images.apply(lambda row: row.imagepath.split("/")[-2], axis=1)


# defining the color of pixels outside of the roi in RGB
mask_color = [0, 0, 0]

# resize ratio for computational speed
resize_ratio = 0.1

# figure size for plotting
figure_size = 7

# figure ids
ids = []
for index, file_id in enumerate(all_cervix_images['file_id']):
    ids.append([file_id, all_cervix_images['type'][index]])

for i in range(len(ids[:])):
    if i%10 == 0:
        print('Loading image %i out of %i' % (i+1, len(ids[:])))
        
    image_id = ids[i]
    image = get_image_data(image_id[0], image_id[1], resize_ratio)
    
    # watershed algorithm
    new_image, boolean_image_mask = crop_roi(image, True)
    fname = "../train_roi/{}/{}.jpg".format(image_id[1],image_id[0])
    plt.imsave(fname, new_image, dpi=600)

