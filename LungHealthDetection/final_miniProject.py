#!/usr/bin/env python
# coding: utf-8

# In[1]:


# to ignore the warnings 
import warnings
warnings.filterwarnings("ignore")


# In[2]:


import scipy
import pandas as pd
import numpy as np
import os
import shutil
import glob
import matplotlib.pyplot as plt 


# In[3]:


from keras.preprocessing.image import ImageDataGenerator
from keras.applications import vgg16
from keras.models import Model
from keras.layers import Dense, MaxPool2D, Conv2D
import keras
import matplotlib.pyplot as plt 


# In[4]:


class_type = {0:'Covid',  1 : 'Normal'}


# In[5]:


from keras.applications.vgg16 import VGG16
from keras.layers import Flatten , Dense, Dropout , MaxPool2D


# In[6]:


## load only the best model 
from keras.models import load_model
model = load_model("C:/Users/Ashish Thakor/Desktop/mini_project/mini_project/LungHealthDetection/bestmodel.h5")


# In[7]:


from keras.preprocessing import image

def get_img_array(img_path):
    """
    Input : Takes in image path as input 
    Output : Gives out Pre-Processed image
    """
    path = img_path
    img = image.load_img(path, target_size=(224,224,3))
    img = image.img_to_array(img)/255
    img = np.expand_dims(img , axis= 0 )
    return img


# In[8]:
'''
--------------------------------------------Not require right now---------------------------------------

# # path for that new image. ( you can take it either from google or any other scource)

# path = "C:/Users/Ashish Thakor/Desktop/mini_project/temporary/mini_project(final)/model/finalcontent/all_images/COVID-1152.png"       # you can add any image path

# #predictions: path:- provide any image from google or provide image from all image folder
# img = get_img_array(path)

# res = class_type[np.argmax(model.predict(img))]
# print(f"The given X-Ray image is of type = {res}")
# print()
# print(f"The chances of image being Covid is : {model.predict(img)[0][0]*100} percent")
# print()
# print(f"The chances of image being Normal is : {model.predict(img)[0][1]*100} percent")

# # to display the image  
# plt.imshow(img[0], cmap = "gray")
# plt.title("input image")
# plt.show()

'''
# In[9]:


import tensorflow as tf


# In[10]:


# this function is udes to generate the heat map of aan image


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


# In[11]:


import matplotlib.cm as cm

from IPython.display import Image, display


# In[12]:


# put the heatmap to our image to understand the area of interest

def save_and_display_gradcam(img_path , heatmap, cam_path="cam.jpg", alpha=0.4):
    """
    img input shoud not be expanded 
    """

    # Load the original image
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    display(Image(cam_path))
    a = plt.imread(cam_path)
    plt.imshow(a)
    plt.title("Heat image")
    plt.show()    



# In[13]:


# function that is used to predict the image type and the ares that are affected by covid


def image_prediction_and_visualization(path,last_conv_layer_name = "block5_conv3", model = model):
    """
    input:  is the image path, name of last convolution layer , model name
    output : returs the predictions and the area that is effected
    """
    img_array = get_img_array(path)

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

    img = get_img_array(path)

    res = class_type[np.argmax(model.predict(img))]
    # print(f"The given X-Ray image is of type = {res}")
    # print()
    c = model.predict(img)[0][0]*100
    # print(f"The chances of image being Covid is : {c} %")
    n = model.predict(img)[0][1]*100
    # print(f"The chances of image being Normal is : {n} %")
    l1 = [res,c,n]
    # print()
    # print("image with heatmap representing the covid spot")

  # function call
    save_and_display_gradcam(path, heatmap)
    return l1
    # print()
    # print("the original input image")
    # print()

    # a = plt.imread(path)
    # plt.imshow(a, cmap = "gray")
    # plt.title("Original image")
    # plt.show()


# In[14]:


# #predictions
# # provide the path of any image from google or any other scource 
# # the path is already defigned above , but you can also provide the path here to avoid scrolling up 

# # for covid image : 
# path = "C:/Users/Ashish Thakor/Desktop/mini_project/temporary/mini_project(final)/model/finalcontent/all_images/Covid-1152.png"
# image_prediction_and_visualization(path)


# In[15]:


# # for normal image : 
# path = "C:/Users/Ashish Thakor/Desktop/mini_project/temporary/mini_project(final)/model/finalcontent/all_images/train_test_split/test/Normal/Normal-55.png"
# image_prediction_and_visualization(path)


# In[ ]:





# In[ ]:





# In[ ]:




