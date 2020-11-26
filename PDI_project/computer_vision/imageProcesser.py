import logging
import cv2
from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab
from skimage.io import imsave, imshow
from PIL import Image 
import numpy as np
import os
import random
import tensorflow as tf
import urllib.request
from ..models import *
from django.core.files.base import ContentFile
from django.conf import settings
from django.contrib.staticfiles import finders


logger = logging.getLogger(__name__)

def LogPrint(object):
    logger.error(": "+str(object))

def ProcessImage(request):
    if request is not None:
        # Grab uploaded image
        image = GrabImage(stream=request)
        # Modify Image

        image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)

        image = np.asarray(image, dtype=float)

        X = 1.0/255*image[:,:,0]
        X = X.reshape(1, 400, 400, 1)

        model = tf.keras.models.load_model(finders.find('trained_models/my_model'))
        output = model.predict(X)    
        output *= 128
        # Output colorizations
        cur = np.zeros((400, 400, 3))
        cur[:,:,0] = X[0][:,:,0]
        cur[:,:,1:] = output[0]
        
        # lab2rgb(cur)
        output = cur.astype("uint8")
        output = cv2.cvtColor(output,cv2.COLOR_Lab2RGB)
        #output = output * 255
        LogPrint(output)
        #output = Image.fromarray(cur.astype(np.uint8))
        #output = cv2.cvtColor(output, cv2.COLOR_Lab2RGB)

        # Save image on disk
        imageURL = SaveImage(output, request.name)
        # Return processed image URL
        return imageURL

def GrabImage(path=None, stream=None, url=None):
    # if the path is not None, then load the image from disk
    if path is not None:
        image = cv2.imread(path)
    # otherwise, the image does not reside on disk
    else:	
        # if the URL is not None, then download the image
        if url is not None:
            resp = urllib.request.urlopen(url)
            data = resp.read()
        # if the stream is not None, then the image has been uploaded
        elif stream is not None:
            data = stream.read()
        # convert the image to a NumPy array and then read it into
        # OpenCV format
        image = np.asarray(bytearray(data), dtype="uint8")
        # Decode into bgr 
        image = cv2.imdecode(image, cv2.IMREAD_COLOR) 
        # Convert into rgb 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # return the image
    return image

def SaveImage(image, filename=None):
    # Encode back
    ret, buf = cv2.imencode('.jpg', image)
    content = ContentFile(buf.tobytes())
    # Create model
    imgModel = ImageModel()
    # Save into disk
    imgModel.image.save(filename,content)
    # Return local image URL
    return imgModel.image.url