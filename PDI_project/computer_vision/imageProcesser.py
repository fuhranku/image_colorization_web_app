import logging
import cv2
import numpy as np
import urllib.request
from ..models import *
from django.core.files.base import ContentFile


logger = logging.getLogger(__name__)

def LogPrint(object):
    logger.error(": "+str(object))

def ProcessImage(request):
    if request is not None:
        # Grab uploaded image
        image = GrabImage(stream=request)
        # Modify Image
        cv2.blur(image,(20,20),image)
        # Save image on disk
        imageURL = SaveImage(image, request.name)
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
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
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