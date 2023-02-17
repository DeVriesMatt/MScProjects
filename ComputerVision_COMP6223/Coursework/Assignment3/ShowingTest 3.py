import numpy as np

patch_test = np.zeros((8, 8, 3))
for i in range(12, 20):
    for j in range(8):
        patch_test[i-12, j] = images[1][i, j]


# I changed the image.extract_patches_2d() function to
# have a step size of 12 thus meaning that 8x8 patches are sampled
# every 4 pixels from the image.
# Probably an easier way...

patches_test = image.extract_patches_2d(images[1], (8, 8))

print(patches_test[1])
print(patch_test == patches_test[24])
