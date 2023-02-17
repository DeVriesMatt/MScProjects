import matplotlib.pyplot as plt
from skimage import data, filters

image = data.coins()   # ... or any other NumPy array!
edges = filters.sobel(image)
plt.imshow(image)
plt.show()
plt.imshow(edges, cmap='gray')
plt.show()
