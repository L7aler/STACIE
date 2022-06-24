#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from skimage.feature import hog
import PIL
from PIL import Image, ImageFilter ,ImageEnhance,ImageOps
from skimage.util import random_noise
import matplotlib.pyplot as plt     
import cv2
from dataset import Solar_Dataset
from model_tens import Solar_Classifier
import random


# In[3]:


def contrast(img, factor=0):
    
    if factor == 0:
        factor = np.random.choice([1.5,2,2.5,3,3.5,4])
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)
    else:
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)
   


# In[4]:


def sharpness(img, factor=0):
    
    if factor == 0:
        factor = np.random.choice([1.5,2,2.5,3,3.5,4])
        enhancer = ImageEnhance.Sharpness(img)
        return enhancer.enhance(factor)
    else:
        enhancer = ImageEnhance.Sharpness(img)
        return enhancer.enhance(factor)
   


# In[5]:


def noise(img, noise_mode = 's&p'):
    img_cv2 = np.asarray(img, 'float32')
    #img_cv2_og = np.asarray(img_og, 'float32')

    noise = random_noise(img_cv2, mode=noise_mode)
    noise_img = img_cv2 * noise
    pil = Image.fromarray((noise_img*255).astype('uint8'))
    return pil


# In[6]:


def flip(img):
    return ImageOps.mirror(img)


# In[7]:


def rotate(img):
    return img.rotate(np.random.choice([0, 90, 180, 270]))


# In[8]:


def edges(img, img_og):
    # Converting the image to grayscale, as edge detection 
    # requires input image to be of mode = Grayscale (L)
    img = img.convert("L")
    # Detecting Edges on the Image using the argument ImageFilter.FIND_EDGES
    img=  img.filter(ImageFilter.FIND_EDGES)
    return img


# In[2]:


def augment_image(img, functions):

    for function in functions:
        img = function(img)
    return img


# In[1]:


def data_aug(dataset):

    functions_pool = [contrast, sharpness, flip, rotate]
    optional_noise = ['s&p', 'gaussian', 'speckle', 'salt', 'pepper']
    for i,img in enumerate(dataset.data):
        img_pil = Image.fromarray((img[0]*255).astype('uint8'))
        no = np.random.choice(np.arange(1,len(functions_pool)+1))
        functions = random.sample(functions_pool, no)

        optional = np.random.choice([True, False])
        aug_img = augment_image(img_pil, functions)
        if optional:
           noise_type = np.random.choice(optional_noise)

           aug_img = noise(aug_img, noise_type)


        dataset.data[i][0] = np.array(aug_img)/255
    return dataset



# In[ ]:




