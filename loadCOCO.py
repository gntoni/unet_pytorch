#!/usr/bin/env python

import numpy as np
import torch
from os import path
from scipy.misc import imread, imresize
from scipy.io import loadmat


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (int or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image, labels):
        assert image.shape[:2] == labels.shape

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = imresize(image, (new_h, new_w))
        lbls = imresize(labels, (new_h, new_w), interp="nearest")

        return (img, lbls)


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image, labels):
        assert image.shape[:2] == labels.shape

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        labels = labels[top: top + new_h,
                        left: left + new_w]

        return (image, labels)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image, labels):
        assert image.shape[:2] == labels.shape

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return (torch.from_numpy(image),
                torch.from_numpy(labels))


def loadCOCO(dataset_folder):
    resc = Rescale(650)
    crop = RandomCrop(640)

    namespath = path.join(dataset_folder, "imageLists/train.txt")
    names = np.loadtxt(namespath, dtype=str, delimiter="\n")

    images = []
    labels = []
    for imgName in names:
        im = imread(path.join(dataset_folder, "images/"+imgName+".jpg"), mode="RGB")
        mat = loadmat(path.join(dataset_folder, "annotations/"+imgName+".mat"))
        lbl = mat["S"]

        im, lbl = resc(im, lbl)
        im, lbl = crop(im, lbl)
        images.append(im)
        labels.append(lbl)

    images = np.array(images, dtype='float32')
    images /= 255.0  # Span 0 ~ 1
    images = (images*2) - 1  # Span -1 ~ 1

    return (images, np.array(labels))


if __name__ == '__main__':
    DATASET_FOLDER = "/home/toni/Data/ssegmentation/COCO"
    loadCOCO(DATASET_FOLDER)
