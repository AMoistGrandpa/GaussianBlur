from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import math

class Image:
    """
    Wrapper class to implement exception raising for negative index access in numpy arrays.
    """
    #TODO: implement support for slices in this wrapper class. Right now I don't think I'll need it but I probably should anyway
    def __init__(self, arr:np.ndarray):
        self.array = arr

    def __getitem__(self, indices):
        if min(indices) < 0:
            raise IndexError('Negative indices are invalid.')
        else:
            return self.array[indices]

    def __setitem__(self, indices, value):
        if min(indices) < 0:
            raise IndexError('Negative indices are invalid.')
        else:
            self.array[indices] = value


class Kernel:

    def __init__(self, data: Optional[np.ndarray] = None):
        if data is None:
            self.data = None
            self.total = None
        else:
            self.data = data
            self.total = data.sum()

    def convolve(self, image:Image) -> Image:
        """
        Applies the filter contained in this kernel.

        :param image: The image to filter.
        :return: The filtered image.
        """
        if self.data is None or self.total is None:
            raise ValueError('Invalid kernel state. Did you initialize the kernel?')

        new_image = np.empty(image.array.shape, np.uint8)
        for y in range(image.array.shape[0]):
            print(y)
            for x in range(image.array.shape[1]):
                new_image[y, x] = self.sumProduct(image, x, y)
        return Image(new_image)

    def sumProduct(self, image:Image, image_x, image_y) -> float:
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
        x_of_leftmost_pixel_in_kernel = image_x - int(kernel_width/2)
        y_of_top_pixel_in_kernel = image_y - int(kernel_height / 2)
        running_total = 0
        for y in range(kernel_height):
            for x in range(kernel_width):
                try:
                    running_total += self.data[y,x] * image[y_of_top_pixel_in_kernel + y, x_of_leftmost_pixel_in_kernel + x]
                except IndexError:
                    continue # I chose to ignore non-existent pixels, but you could easily add code here to get data from the other side of the image or something
        return running_total/self.total # normalize


def generateSeparatedGaussianKernel(std_dev) -> Kernel:
    """
    Creates a kernel suitable for using to perform a Gaussian blur. The generated kernel will have dimensions
    ceil(6*sigma) x 1. To perform a blur you will have to convolve this kernel with the image twice: once as it is and
    once with the kernel transposed on itself.

    :param std_dev:
        The standard deviation of the Gaussian distribution used in calculating the constants in the kernel.
        Lower numbers mean fewer pixels are used in each calculation, which means less blurring.
        The radius of the blur is equal to ceil(3*std_dev).
    :return: A populated kernel with data suitable for performing Gaussian blurs.
    """
    height = 1
    width = int(np.ceil(std_dev * 6))
    if width % 2 == 0:
        width += 1 # I haven't tried it but I think my code breaks with a kernel with any dimension being an even number
    tempArr = np.empty((height, width), dtype=np.float64)

    for x in range(int(-width/2), int(width/2) + 1):
        tempArr[0, x + int(width/2)] = (1/math.sqrt(2*math.pi*std_dev**2))*math.exp(-(x**2)/(2*std_dev**2))

    return Kernel(tempArr)

img = plt.imread('flower.jpg')
plt.imshow(img)
plt.axis('off')
plt.show()


k = generateSeparatedGaussianKernel(3)

r = Image(img[:,:,0])
g = Image(img[:,:,1])
b = Image(img[:,:,2])
print('starting')
r = k.convolve(r)
k.data = k.data.T
r = k.convolve(r)
print('r done')
g = k.convolve(g)
k.data = k.data.T
g = k.convolve(g)
print('g done')
b = k.convolve(b)
k.data = k.data.T
b = k.convolve(b)
print('b done')


r = r.array
g = g.array
b = b.array

new_img = np.array([r.T, g.T, b.T]).T
plt.imshow(new_img)
plt.axis('off')
plt.show()
print()
