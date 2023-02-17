import logging
import os

from skimage.io import imread

ABSOLUTE_PATH = os.path.dirname(__file__)


def load_labelled(path):
  """
  Load and return the labelled training data.

  :param path: str, Path to labelled images (e.g '../../data/training')
  :return: (data, target) : tuple
  """
  path = f'{ABSOLUTE_PATH}/{path}'

  if os.path.exists(path):
    labels = os.listdir(path)
    x = []
    y = []

    logging.info(f'Loading labelled data from {path}.')
    logging.info(f'{"Class name":20}{"Number of samples":20}')
    logging.info(f'{"".join(["-"] * 40)}')
    for label in labels:
      images = [imread(f'{path}/{label}/{image}') for image in
                os.listdir(f'{path}/{label}')]
      x.extend(images)
      y.extend([label] * len(images))
      logging.info(f'{label:20}{len(images):<20}')

    logging.info(
      f'Loaded {len(labels)} classes. Total number of samples is {len(x)}.')

    return x, y
  else:
    raise FileExistsError(f'The path "{path}" does not exist.')


def load_unlabelled(path):
  """
  Load and return the unlabelled testing data.
  :param path: str, Path to unlabelled images (e.g '../../data/testing')
  :return: data : list
  """
  path = f'{ABSOLUTE_PATH}/{path}'

  if os.path.exists(path):
    x = []

    logging.info(f'Loading unlabelled data from {path}.')
    images = [imread(f'{path}/{image}') for image in os.listdir(f'{path}')]
    x.extend(images)
    logging.info(f'Total number of samples is {len(x)}.')

    return x
  else:
    raise FileExistsError(f'The path "{path}" does not exist.')
