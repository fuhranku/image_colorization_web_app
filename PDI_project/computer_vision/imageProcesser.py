import logging
import cv2
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image
from skimage.transform import resize 
import numpy as np
import os
import random
import tensorflow as tf
import urllib.request
from ..models import *
from django.core.files.base import ContentFile
from django.conf import settings
from django.contrib.staticfiles import finders

def LogPrint(object):
    logger = logging.getLogger(__name__)
    logger.error(": "+str(object))

def ProcessImage(request, option=1):
    if request is not None:
        sizes = [(400,400),(256,256)]
        # Grab uploaded image
        image = LoadImage(stream=request)
        # Get dimensions of image
        (H,W) = image.shape[:2]
        # Resize to fit model (256x256)
        image = Resize(image, *sizes[option])
        # Modify Image
        image = ColorizeAlpha(image) if option == 0 else ColorizeFull(image)
        # Resize image to original size
        image = Resize(image, *(W,H))
        # Save image on disk
        imageURL = SaveImage(image, request.name)
        # Return processed image URL
        return imageURL

def Resize(image, W,H):
    return  cv2.resize(image,(W,H),interpolation=cv2.INTER_CUBIC)


#Create embedding
def create_inception_embedding(grayscaled_rgb, inception):
    grayscaled_rgb_resized = []
    for i in grayscaled_rgb:
        i = resize(i, (299, 299, 3), mode='constant')
        grayscaled_rgb_resized.append(i)
    grayscaled_rgb_resized = np.array(grayscaled_rgb_resized)
    grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)
    with inception.graph.as_default():
        #Load weights
        inception.load_weights(finders.find('trained_models/resnet/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5'))
        embed = inception.predict(grayscaled_rgb_resized)
    return embed

def ColorizeAlpha(image):
    # Get dimensions of image
    (H,W) = image.shape[:2]
    # Normalize image to handle variations in intensity
    image = (image * 1.0 / 255).astype(np.float32)
    # Convert to Lab color space and grab channel L
    image = cv2.cvtColor(image,cv2.COLOR_RGB2Lab)
    # Get L channel from image
    X = image[:,:,0]
    # Resize image to network input size
    X = X.reshape(1, 400, 400, 1)
    # Load model
    model = tf.keras.models.load_model(finders.find('trained_models/my_model'))
    # Colorize
    output = model.predict(X)
    # Denormalize ab channels from L * a * b
    output *= 128
    # Putting all together
    colorizedImage = np.zeros((W,H,3))
    # Get L Channel from prepros image
    colorizedImage[:,:,0] = X[0][:,:,0]
    # Get ab Channel from output image
    colorizedImage[:,:,1:] = output[0]
    # Convert to float32
    colorizedImage = colorizedImage.astype(np.float32)
    # Convert to RGB
    colorizedImage = cv2.cvtColor(colorizedImage,cv2.COLOR_Lab2RGB)
    # Denormalize values
    colorizedImage = colorizedImage * 255
    # Cast values to unsigned int
    colorizedImage = colorizedImage.astype(np.uint8)

    return colorizedImage

def ColorizeFull(image):
    tf.compat.v1.disable_v2_behavior()
    # Get dimensions of image
    (H,W) = image.shape[:2]
    # Normalize image to handle variations in intensity
    image = (image * 1.0 / 255).astype(np.float32)
    # Convert to Lab color space and grab channel L
    image = cv2.cvtColor(image,cv2.COLOR_RGB2Lab)
    # Get L channel from image
    X = image[:,:,0]
    # Resize image to network input size
    X = X.reshape(1,W,H,1)
    # Load trained model
    model = tf.keras.models.load_model(finders.find('trained_models/full_model'))

    inception = InceptionResNetV2(weights=None, include_top=True)
    inception.graph = tf.compat.v1.get_default_graph()
    X_embed = create_inception_embedding(X, inception)
    # Colorize Image with CNN
    output = model.predict([X,X_embed])
    # Denormalize ab channels from L * a * b
    output *= 128
    # Putting all together
    colorizedImage = np.zeros((W,H,3))
    # Get L Channel from prepros image
    colorizedImage[:,:,0] = X[0][:,:,0]
    # Get ab Channel from output image
    colorizedImage[:,:,1:] = output[0]
    # Convert to float32
    colorizedImage = colorizedImage.astype(np.float32)
    # Convert to RGB
    colorizedImage = cv2.cvtColor(colorizedImage,cv2.COLOR_Lab2RGB)
    # Denormalize values
    colorizedImage = colorizedImage * 255
    # Cast values to unsigned int
    colorizedImage = colorizedImage.astype(np.uint8)

    return colorizedImage

def LoadImage(path=None, stream=None, url=None):
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
        # Decode into BGR 
        image = cv2.imdecode(image, cv2.IMREAD_COLOR) 
        # Convert into RGB 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # return the image
    return image

def SaveImage(image, filename=None):
    # Convert back into BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Encode back
    ret, buf = cv2.imencode('.jpg', image)
    content = ContentFile(buf.tobytes())
    # Create model
    imgModel = ImageModel()
    # Save into disk
    imgModel.image.save(filename,content)
    # Return local image URL
    return imgModel.image.url
