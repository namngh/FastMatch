import cv2
import numpy as np
import math


def generate_rotate_matrix(rotate):
    return np.matrix([[math.cos(rotate), -math.sin(rotate)], [math.sin(rotate), math.cos(rotate)]])


def preprocess_image(image):
    image_clone = image.copy()

    if len(image_clone.shape) == 3:
        image_clone = cv2.cvtColor(image_clone, cv2.COLOR_BGR2GRAY)

    height, width = image_clone.shape

    height = height if height % 2 == 0 else height - 1
    width = width if width % 2 == 0 else width - 1

    return image_clone[0:height, 0:width]


def validate_affine_corner(point, top_left, bottom_right):
    return point[0] > top_left[0] and point[1] > top_left[1] and point[0] < bottom_right[0] and point[1] < bottom_right[1]
