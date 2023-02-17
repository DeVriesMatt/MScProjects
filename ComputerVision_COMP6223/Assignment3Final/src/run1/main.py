import logging

from src.misc.data import load_labelled, load_unlabelled
from src.misc.visualise import draw_labelled_samples


def main():
  # Set up logging.
  logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

  # Load training data.
  x, y = load_labelled('../../data/training')

  # Load testing data.
  x_test = load_unlabelled('../../data/testing')

  # Visualise the data.
  draw_labelled_samples(x, y)


if __name__ == '__main__':
  main()
