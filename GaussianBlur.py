import numpy as np
import matplotlib.pyplot as plt

class Image:
    """
    Wrapper object to implement exception raising for negative index access in numpy arrays.
    """
    #TODO: implement support for slices in this wrapper class. Right now I don't think I'll need it but I probably should anyway
    def __init__(self, arr):
        self.arr = arr

    def __getitem__(self, *indices):
        if min(indices) < 0:
            raise IndexError('Negative indices are invalid.')
        else:
            return self.arr[indices]

    def __setitem__(self, *indices): # would have a value param but that breaks stuff
        if min(indices[:-1]) < 0:
            raise IndexError('Negative indices are invalid.')
        else:
            if len(indices) < 2:
                raise TypeError('This error should never be raised.')
            self.arr[indices[:-1]] = indices[-1]


img = plt.imread('flower.jpg')
plt.imshow(img)
plt.axis('off')

x = np.array([1,2,3])
x = Image(x)
x[-1] = 5
print(x)