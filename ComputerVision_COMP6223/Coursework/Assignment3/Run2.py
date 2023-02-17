import glob
import cv2
import numpy as np
from sklearn.feature_extraction import image
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn import linear_model

print('starting')

folders = glob.glob('training/*')

image_names_list = []
for folder in folders:
    for f in glob.glob(folder+'/*.jpg'):
        image_names_list.append(f)

read_images = []

for img in image_names_list:
    read_images.append(cv2.imread(img))



# Function for getting the fixed size densely-sampled pixel patches
# def pixel_patches(image_list, size):
#     # for every image there are a number of patches of shape (8,8,3)
#
#     sampled_patches = []
#     for im in range(len(image_list)):
#         patch = image.extract_patches_2d(image_list[im], (size, size))
#         sampled_patches.append(patch)
#
#     return sampled_patches
#
#
# samp_patches = pixel_patches(read_images, 8)
# # we need to flatten every patch into a vector
# flattened = []
# for img in range(len(samp_patches)):
#     for p in range(samp_patches[img].shape[0]):
#         flattened.append(samp_patches[img][p].flatten())
#
#
# # The resulting vector is made from x,y,color, so we take every this element to make gray
# flat = []
# for p in range(len(flattened)):
#     v = flattened[p][::3]
#     flat.append(v)


# Standardize each vector
for p in range(len(flat)):
    if np.std(flat[p]) == 0:
        flat[p] = (flat[p] - np.mean(flat[p]))
    else:
        flat[p] = (flat[p] - np.mean(flat[p])) / np.std(flat[p])

# make into array for clustering
training_data = np.array(flat)

print(training_data.shape)

# make sure no na
print(np.argwhere(np.isnan(training_data)))

# Take a random sample of size n
n = 100000
sample_df = training_data[np.random.randint(training_data.shape[0], size=n), :]

# K Means clustering
k_means = KMeans(n_clusters=500, random_state=0).fit(sample_df)
centroids = k_means.cluster_centers_


for image in read_images:
    global_image_feature = np.zeros(500)
    for patch in pixel_patches([read_images[1]], 8):
        patch_vector = patch.flatten()
        c = k_means.predict(patch_vector[::3])
        global_image_feature[c] += 1

# plt.hist(global_image_feature, bins=500)
# plt.show()
print(global_image_feature)


clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
clf.fit(global_image_feature, Y)


def build_histogram(descriptor_list, cluster_alg):
    histogram_ = np.zeros(len(cluster_alg.cluster_centers_))
    cluster_result = cluster_alg.predict(descriptor_list)
    for i in cluster_result:
        histogram_[i] += 1.0
    return histogram_

