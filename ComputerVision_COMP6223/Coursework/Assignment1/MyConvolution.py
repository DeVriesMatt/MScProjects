import numpy as np


def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve an image with a kernel assuming zero-padding of the image to handle the borders

    :param image: the image (either greyscale shape=(rows,cols) or colour shape=(rows,cols,channels))
    :type numpy.ndarray

    :param kernel: the kernel (shape=(kheight,kwidth); both dimensions odd)
    :type numpy.ndarray

    :returns the convolved image (of the same shape as the input image)
    :rtype numpy.ndarray
    """
    # Your code here. You'll need to vectorise your implementation to ensure it runs
    # at a reasonable speed.
    # kernel is a template

    # first we need to flip the kernel along the x and y axis
    kernel = np.flipud(np.fliplr(kernel))
    kernel_r, kernel_c = kernel.shape
    # we define the padding parameters
    pad_r, pad_c = (kernel_r // 2), (kernel_c // 2)
    # the output image must be the same size as the input image
    output_image = np.zeros(image.shape)

    # We need to do a check to see if the image is RBG or not
    if len(image.shape) == 3:
        for color in range(image.shape[2]):
            # make padded image for every level
            level = image[:, :, color]
            # it's also possible to pad a 2D numpy arrays by passing a tuple
            # of tuples as padding width, which takes the format of ((top, bottom), (left, right)):
            padded_image = np.lib.pad(level, ((pad_r, pad_r), (pad_c, pad_c)))
            # Slide kernel across every pixel.
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    # we need to do point by point multiplication
                    output_image[i, j, color] = (kernel * padded_image[i:i + kernel_r, j:j + kernel_c]).sum()

    if len(image.shape) == 2:
        padded_image = np.lib.pad(image, ((pad_r, pad_r), (pad_c, pad_c)))
        # Slide kernel across every pixel.
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                # we need to do point by point multiplication
                output_image[i, j] = (kernel * padded_image[i:i + kernel_r, j:j + kernel_c]).sum()

    # Return the result of the convolution.
    return output_image
