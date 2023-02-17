import math

import cv2 as cv


def crop_to_square(image):
  """
  Crops an image about the centre to the lower dimension.
  :param image: np.array : The b&w image to be cropped
  :return: np.array : The cropped image
  """
  w, h = image.shape
  min_dimension = min(w, h)
  w_start = (w - min_dimension) // 2
  h_start = (h - min_dimension) // 2
  return image[w_start:w_start + min_dimension, h_start:h_start + min_dimension]


def resize(image, size, interpolation=cv.INTER_AREA):
  """
  Resize an image by keeping the aspect ratio.
  :param image: The image to resize.
  :param size: The new size of the smaller dimension in pixels.
  :param interpolation: OpenCV interpolation method.
  :return: The resized image.
  """
  h, w = image.shape
  min_dimension = min(w, h)
  scaling = min_dimension / size

  new_dimension = (
    math.ceil(round(h / scaling, 5)), math.ceil(round(w / scaling, 5)))

  resized = cv.resize(image, new_dimension, interpolation=interpolation)

  return resized
