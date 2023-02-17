from MyHybridImages import myHybridImages
import numpy as np
from PIL import Image
from skimage.transform import resize
from MyConvolution import convolve

np.set_printoptions(suppress=True)

cat = np.array(Image.open("data/cat.bmp"))
dog = np.array(Image.open("data/dog.bmp"))
# high_cat = np.array(Image.open("data/high_frequencies.bmp"))
# low_dog = np.array(Image.open("data/low_frequencies.bmp"))
# notes_hybrid = np.array(Image.open("data/cat_hybrid_image_scales.bmp"))

# einstein = np.array(Image.open("data/einstein.bmp"))
# marilyn = np.array(Image.open("data/marilyn.bmp"))
donald = np.array(Image.open("data/Donald.bmp"))
kim = np.array(Image.open("data/kim_smile.bmp"))
brent = np.array(Image.open("data/hybrid_creation.bmp"))
# human = np.array(Image.open("data/human.bmp"))
# robot = np.array(Image.open("data/robot.bmp"))
# hybrid = myHybridImages(donald, 3, kim, 4)
# hybrid1 = resize(hybrid, (202, 190, 3))
# hybrid2 = resize(hybrid, (101, 95, 3))
# hybrid3 = resize(hybrid, (50, 47, 3))
# hybrid4 = resize(hybrid, (24, 25, 3))

#

test1 = myHybridImages(dog, 6, cat, 8)
img = Image.fromarray(test1)
img.show()

print(test1 == brent)




