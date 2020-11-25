import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

def log_print(num):
    logger.error("Logging: "+str(num))
