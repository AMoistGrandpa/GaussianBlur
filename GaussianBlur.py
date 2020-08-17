from typing import Union, Any
import numpy as np
import matplotlib.pyplot as plt

class Image:
    """
    Wrapper class to implement exception raising for negative index access in numpy arrays.
    """
    #TODO: implement support for slices in this wrapper class. Right now I don't think I'll need it but I probably should anyway
    def __init__(self, arr:np.ndarray):
        self.array = arr

    def __getitem__(self, *indices):
        if min(indices) < 0:
            raise IndexError('Negative indices are invalid.')
        else:
            return self.array[indices]

    def __setitem__(self, *indices): # would have a value param but that breaks stuff
        if min(indices[:-1]) < 0:
            raise IndexError('Negative indices are invalid.')
        else:
            if len(indices) < 2:
                raise TypeError('This error should never be raised.')
            self.array[indices[:-1]] = indices[-1]

#class SquareKernel:
#    def __init__(self, data:Union[np.ndarray, None] = None):
#        if data is None:
#            self.size = 0
#            self.data = None
#            self.total = None
#        else:
#            if data.shape[0] != data.shape[1]:
#                raise ValueError('data array must be square')
#            else:
#                self.size = data.shape[0]
#                self.data = data
#                self.total = sum(sum(data)) # this should be a 2d array. Each sum() reduces the number of dimensions by one
#
#
#    def convolve(self, imgArr):
#        pass

class SeparatedKernel:

    def __init__(self, data: Union[np.ndarray, None] = None):
        if data is None:
            self.size = 0
            self.data = None
            self.total = None
        else:
            if len(data.shape) == 1:
                self.size = data.shape[0]
                self.data = data
                self.total = sum(data)
            else:
                raise ValueError('data array must be 1 dimensional')

    def convolve(self, image:Image):
        """
        Applies the filter contained in this kernel.
        :param image: The image to filter.
        :return: The filtered image.
        """
        new_image = np.empty(image.array.shape, np.uint8)
        for y in range(image.array.shape[0]):
            for x in range(image.array.shape[1]):
                new_image[y, x] = self.sumProduct(image, x, y)
        return new_image

    def sumProduct(self, image:Image, image_x, image_y):
        """
        Helper function for convolve().
        Applies one step in the convolution process. Given a location on the image, multiplies each pixel with its
        corresponding constant in the kernel, then adds all those results.
        :param image: The image that is being convolved.
        :param image_x: The x position of the center of the kernel.
        :param image_y: The y position of the center of the kernel.
        :return: The sum of the products of the kernel and the image pixel intensities.
        """
        kernel_height = self.data.shape[0]
        kernel_width = self.data.shape[1]
        x_of_leftmost_pixel_in_kernel = image_x - (kernel_width - 1)/2
        y_of_top_pixel_in_kernel = image_y - (kernel_height - 1) / 2
        total = 0
        for y in range(kernel_height):
            for x in range(kernel_width):
                try:
                    total += self.data[y,x] * image[y_of_top_pixel_in_kernel + y, x_of_leftmost_pixel_in_kernel + x]
                except IndexError:
                    continue
        return total




img = plt.imread('flower.jpg')
plt.imshow(img)
plt.axis('off')
plt.show()

