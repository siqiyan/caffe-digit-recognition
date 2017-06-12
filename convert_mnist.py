#!/usr/bin/env python

"""
This script apply data augmentation on MNIST dataset, and convert each digit
into single image.
"""

import numpy as np
import cv2
import struct
import random
import os
from scipy.misc import imresize

image_h = 28
image_w = 28

train_size = 50000
test_size = 5000

join = lambda x, y: os.path.join(x, y)

def load_utyte(root, images_ubyte, labels_ubyte):
    print 'load %s ...'%(images_ubyte)
    with open(join(root, images_ubyte), 'rb') as f:
        f.seek(4)
        num = f.read(4)
        f.seek(16)
        ubyte = f.read()
    num = struct.unpack('>I', num)[0]
    assert len(ubyte) / num == 784
    data = np.zeros([num,784], dtype=np.uint8)
    for i in xrange(num):
        for j in xrange(784):
            data[i, j] = struct.unpack('@B', ubyte[j+784*i])[0]
    data = np.reshape(data, [num, 28, 28])

    print 'load %s ...'%(labels_ubyte)
    with open(join(root, labels_ubyte), 'rb') as f:
        f.seek(4)
        num = f.read(4)
        f.seek(8)
        ubyte = f.read()
    num = struct.unpack('>I', num)[0]
    assert len(ubyte) == num
    labels = []
    for i in xrange(num):
        labels.append(struct.unpack('@B', ubyte[i])[0])
    return data, labels

def change_ratio(img):
    # assert len(img.shape) == 2
    # h, w = img.shape
    # pts1 = np.float32([[0,0], [w, 0], [0, h]])
    # offset = random.uniform(0, h * 0.2)
    # pts2 = np.float32([[0,offset], [w,offset], [0,h-offset]])
    # M = cv2.getAffineTransform(pts1, pts2)
    # img = cv2.warpAffine(img, M, (w, h))
    return img

def random_shift(img):
    assert len(img.shape) == 2
    h, w = img.shape
    x_max = w * 0.2
    y_max = h * 0.2
    dx = random.uniform(-x_max, x_max)
    dy = random.uniform(-y_max, y_max)
    M = np.float32([[1,0,dx],[0,1,dy]])
    img = cv2.warpAffine(img, M, (w, h))
    return img

def add_black_edges(img):
    """
    Reduce the effect of black edges around the image
    Note: currently the black has value 255, because in the final statge the
        color will be inverted and the black will be 0
    """
    assert len(img.shape) == 2
    h, w = img.shape
    pts1 = np.float32([[0,0], [w, 0], [0, h]])
    offset = random.uniform(0, h * 0.1)
    pts2 = np.float32([[offset,offset], [w-offset,offset], [offset,h-offset]])
    M = cv2.getAffineTransform(pts1, pts2)
    img = cv2.warpAffine(img, M, (w, h), borderValue=255)
    return img

def random_rotate(img):
    assert len(img.shape) == 2
    h, w = img.shape
    theta = (random.random() * 2 - 1) * 8 # range (-8, 8) degree
    M = cv2.getRotationMatrix2D((h/2, w/2), theta, 1)
    img = cv2.warpAffine(img, M, (w, h), borderValue=255)
    return img

def change_intensity(img):
    img = img.astype(np.int32) + np.random.randint(-80, 80)
    img[img < 0] = 0
    img[img > 255] = 255
    img = img.astype(np.uint8)
    return img

def add_noise(img):
    noise = np.random.normal(0, 25.0, [image_h, image_w])
    img = img.astype(np.float32) + noise
    img[img < 0] = 0.0
    img[img > 255] = 255.0
    return img.astype(np.uint8)

Transformations = [
        # change_ratio,
        random_shift,
        add_black_edges,
        random_rotate,
        change_intensity,
        add_noise
        ]

def generate_images(data, labels, data_size, name):
    root = 'dataset'
    dest = join(root, name)
    flabels = open(join(root, name + '.list'), 'w')
    for im in xrange(data_size):

        select = random.randint(0, data.shape[0] - 1)
        digit = data[select, ...]
        label = labels[select]
        digit = imresize(digit, [image_h, image_w])

        for trans in Transformations:
            digit = trans(digit)
        digit = 255 - digit

        output_name = '%d.jpg'%(im)
        cv2.imwrite(join(dest, output_name), digit)
        flabels.write('%s %d\n'%(join(name, output_name), label))

        print '%s (%d/%d) processed: %s'%(name, im, data_size, output_name)
    flabels.close()

data, labels = load_utyte('MNIST', 'train-images-idx3-ubyte', 'train-labels-idx1-ubyte')
generate_images(data, labels, train_size, 'mnist_aug_train')

data, labels = load_utyte('MNIST', 't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')
generate_images(data, labels, test_size, 'mnist_aug_test')
