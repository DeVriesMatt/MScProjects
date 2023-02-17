import math
import numpy as np
from MyConvolution import convolve


def myHybridImages(lowImage: np.ndarray, lowSigma: float, highImage: np.ndarray, highSigma: float) -> np.ndarray:
    """
    Create hybrid images by combining a low-pass and high-pass filtered pair.

    :param lowImage: the image to low-pass filter (either greyscale shape=(rows,cols) or colour shape=(rows,cols,channels))
    :type numpy.ndarray

    :param lowSigma: the standard deviation of the Gaussian used for low-pass filtering lowImage
    :type float

    :param highImage: the image to high-pass filter (either greyscale shape=(rows,cols) or colour shape=(rows,cols,channels))
    :type numpy.ndarray

    :param highSigma: the standard deviation of the Gaussian used for low-pass filtering highImage before subtraction to create the high-pass filtered image
    :type float

    :returns returns the hybrid image created
           by low-pass filtering lowImage with a Gaussian of s.d. lowSigma and combining it with
           a high-pass image created by subtracting highImage from highImage convolved with
           a Gaussian of s.d. highSigma. The resultant image has the same size as the input images.
    :rtype numpy.ndarray
    """
    low_pass = convolve(lowImage, makeGaussianKernel(lowSigma))
    high_pass = highImage - convolve(highImage, makeGaussianKernel(highSigma))
    # We need to threshold the image.
    # To avoid different roundings of different displaying functions, I have taken the floor of each
    # element in the output.
    # Pillow, the library I used, needed the ndarray to be of type uint8.
    return np.uint8(np.clip(high_pass + low_pass, 0, 255))


def makeGaussianKernel(sigma: float) -> np.ndarray:
    """
    Use this function to create a 2D gaussian kernel with standard deviation sigma.
    The kernel values should sum to 1.0, and the size should be floor(8*sigma+1) or
    floor(8*sigma+1)+1 (whichever is odd) as per the assignment specification.
    """

    if math.floor(8*sigma+1) % 2 == 1:
        size = math.floor(8*sigma+1)
    else:
        size = math.floor(8*sigma+1)+1

    half_size = math.floor(size/2)
    xx = np.linspace(-half_size, half_size, size)
    yy = np.linspace(-half_size, half_size, size)
    kernel = np.empty([size, size])
    for i in range(size):
        for j in range(size):
            kernel[i, j] = math.exp(-((xx[i]**2 + yy[j]**2)/(2 * (sigma**2))))

    return kernel/kernel.sum()
