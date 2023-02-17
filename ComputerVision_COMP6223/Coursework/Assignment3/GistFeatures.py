import gist
import numpy as np
import cv2
import glob


folders = glob.glob('training/*')

image_names_list = []
for folder in folders:
    for f in glob.glob(folder+'/*.jpg'):
        image_names_list.append(f)

read_images = []

for img in image_names_list:
    read_images.append(cv2.imread(img))

gist_test = gist.extract(read_images[0])
