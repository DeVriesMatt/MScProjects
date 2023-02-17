import math

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  :param y_true:
  :param y_pred:
  :param classes:
  :param normalize:
  :param title:
  :param cmap:
  :return:
  """
  if not title:
    if normalize:
      title = 'Normalized confusion matrix'
    else:
      title = 'Confusion matrix, without normalization'

  # Compute confusion matrix
  cm = confusion_matrix(y_true, y_pred)
  # Only use the labels that appear in the data
  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

  fig, ax = plt.subplots(figsize=(13, 13))
  im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
  ax.figure.colorbar(im, ax=ax)
  # We want to show all ticks...
  ax.set(xticks=np.arange(cm.shape[1]),
         yticks=np.arange(cm.shape[0]),
         # ... and label them with the respective list entries
         xticklabels=classes, yticklabels=classes,
         title=title,
         ylabel='True label',
         xlabel='Predicted label')

  # Rotate the tick labels and set their alignment.
  plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
           rotation_mode="anchor")

  # Loop over data dimensions and create text annotations.
  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
      ax.text(j, i, format(cm[i, j], fmt),
              ha="center", va="center",
              color="white" if cm[i, j] > thresh else "black")
  fig.tight_layout()
  return ax


def draw_labelled_samples(x, y):
  """

  :param x:
  :param y:
  :return:
  """
  labels = set(y)
  n_rows = math.ceil(len(labels) / 5)

  fig = plt.figure(figsize=(13.5, 7))
  axs = []

  for i in range(n_rows):
    for j in range(5):
      axs.append(plt.subplot2grid((3, 5), (i, j), 1, 1))
      axs[-1].axis('off')

  for i, label in enumerate(labels):
    axs[i].imshow(x[y.index(label)], cmap='gray')
    axs[i].set_title(label)

  fig.tight_layout()
  plt.show()
