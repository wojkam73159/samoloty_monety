from pylab import *
import skimage
from skimage import data, io, filters, exposure, feature
from skimage.filters import rank
from skimage.util.dtype import convert
from skimage import img_as_float, img_as_ubyte
# from skimage.io import Image##########################
# from skimage.viewer import ImageViewer #newer
from skimage.color import rgb2hsv, hsv2rgb, rgb2gray
from skimage.filters.edges import convolve
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy import ndimage as ndi
from numpy import array

from skimage.segmentation import watershed


def int2double_digit_int(x: int) -> chr:
    if (x < 10):
        return '0' + str(x)
    else:
        return str(x)


def immage_summary(image):
    image = img_as_float(image)
    figure(figsize=(20, 20))
    fig, ax = plt.subplots(nrows=1, ncols=1)
    matplotlib.pyplot.hist(image, bins=255)
    # histo, x = np.histogram(img, range(0, 256), density=True)
    # fig= np.histogram(img, range(0, 256), density=True)
    # plot(fig)
    xlim(0, 255)
    plt.show()


def immage_filter(image):
    img2 = filters.sobel(image)
    #    a=img2.shape()
    for i in range(0, img2.shape[0]):
        for j in range(0, img2.shape[1]):
            if (img2[i, j, 0] < 0.09 or img2[i, j, 1] < 0.11 or img2[i, j, 2] < 0.09):
                temp = [0, 0, 0]
                img2[i, j] = tuple(temp)
            if (img2[i, j, 1] - img2[i, j, 0] > 0.07):
                temp = [0, 0, 0]
                img2[i, j] = tuple(temp)
    return img2


def immage_filter_2(image):
    img2 = filters.sobel(image)
    #    a=img2.shape()
    return img2



def immage_denoise(image):
    img2 = filters.sobel(image)
    img2 = img_as_ubyte(img2)
    return filters.rank.mean(img2, ones([3, 3], dtype=uint8))


def immage_normalise(image):
    pass


def immage_stretch_contrast(image):
    img2 = filters.sobel(image)
    #    a=img2.shape()
    MIN = 100 / 256
    MAX = 125 / 256

    norm = (image - MIN) / (MAX - MIN)
    norm[norm > 1] = 1
    norm[norm < 0] = 0
    return norm





if __name__ == '__main__':

    fig, axes = plt.subplots(2, 3, figsize=(20, 20))  # wykresy
    ax = axes.ravel()

    coins = data.coins()
    ax[0].imshow(coins, cmap=plt.cm.gray)
    ax[0].axis('off')

    markers = np.zeros_like(coins)
    markers[coins < 30] = 1
    markers[coins > 150] = 2
    ax[1].imshow(markers, cmap=plt.cm.gray)
    ax[1].axis('off')

    elevation_map = filters.sobel(coins)

    markers = np.zeros_like(coins)
    markers[coins < 30] = 1
    markers[coins > 150] = 2

    segmentation = watershed(elevation_map, markers)

    segmentation = ndi.binary_fill_holes(segmentation - 1)
    ax[2].imshow(segmentation, cmap=plt.cm.gray)
    ax[2].axis('off')

    labeled_coins, two = ndi.label(segmentation)
    ax[3].imshow(labeled_coins)  # , cmap=plt.cm.gray)
    ax[3].axis('off')
    # teraz trzeba kazdy znaleziony z 25 segmentow zaadresowac i pokoloorwac w orginale
    # for i in
    # ax[4].imshow(two, )
    # ax[4].axis('off')

    # skimage.io.imsave('samoloty_saved/moneta-save.jpg', ndi.label(segmentation))

    fig.tight_layout()
    plt.show()

    img = img_as_float(io.imread('samoloty/samolot{}.jpg'.format(int2double_digit_int(0))))
    img2 = img
    img2 = immage_filter(img2)
    img2 = rgb2gray(img2)
    thresh = 0.1
    binary = (img2 > thresh) * 255
    skimage.io.imsave('samoloty_saved/samolot-{}-save.jpg'.format(int2double_digit_int(0)), binary)
    # immage_summary(img)

    # Load images as greyscale but make main RGB so we can annotate in colour
    seg = io.imread('samoloty_saved/samolot-{}-save.jpg'.format(int2double_digit_int(0)))  # , cv2.IMREAD_GRAYSCALE)
    main = io.imread('samoloty/samolot{}.jpg'.format(int2double_digit_int(0)))  # , cv2.IMREAD_GRAYSCALE)
    # main = cv2.cvtColor(main, cv2.COLOR_GRAY2BGR)

    # Create structuring element that defines the neighbourhood for morphology
    selem = skimage.morphology.disk(1)

    # Mask for edges of segment 1 and segment 2
    # We are basically looking for pixels with value 1 in the segmented image within a radius of 1 pixel of a black pixel...
    # ... then the same again but for pixels with a vaue of 2 in the segmented image within a radius of 1 pixel of a black pixel
    seg1 = (skimage.filters.rank.minimum(seg, selem) == 0) & (skimage.filters.rank.maximum(seg, selem) == 1)
    seg2 = (skimage.filters.rank.minimum(seg, selem) == 0) & (skimage.filters.rank.maximum(seg, selem) == 2)

    main[seg1, :] = np.asarray([0, 0, 255])  # Make segment 1 pixels red in main image
    main[seg2, :] = np.asarray([0, 255, 255])  # Make segment 2 pixels yellow in main image

    # Save result
    skimage.io.imsave('help/samolot-{}-save.jpg'.format(int2double_digit_int(0)), binary)












