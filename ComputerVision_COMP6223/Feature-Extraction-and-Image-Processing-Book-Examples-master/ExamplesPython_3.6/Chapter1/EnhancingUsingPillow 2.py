from PIL import Image, ImageFilter
from PIL import ImageEnhance

# Read image
im = Image.open('image.jpg')
# Display image
im.show()

enh = ImageEnhance.Contrast(im)
enh.enhance(1.8).show("30% more contrast")
