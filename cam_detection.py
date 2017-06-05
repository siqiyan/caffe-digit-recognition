#!/usr/bin/env python
"""
This script run the real-time facial emotion recognition using the pre-trained
model. It can take input from either webcam or image file
"""

import cv2
import caffe
import numpy as np
from scipy.misc import imresize
import time
import os
# from include import *


# Change the following parameters as needed:
mode                = 'GPU'
hist_equ            = False
blur                = False
kernel              = (5, 5)
show_img_patch      = False
show_middle_layers  = False
options             = 3
cam_id              = 0
score_decay         = 0.15
image_h             = 28
image_w             = 28

# If you want to run this script on a single image rather than camera, change
# this to the path of the image:
# using_image         = 'test/2.png'
using_image         = None

join = lambda x, y: os.path.join(x, y)

model_def = 'lenet_train_test_deploy.prototxt'
model_weights = join('weights', 'lenet.caffemodel')
# model_mean = 'deploy/kdef_train_mean.binaryproto'



def merge_map(src):
    """
    Merge a multiple channel image into a bigger image with one channel for
    easy displaying.

    Input shape: (ch, height, width)
    Output shape: (Height, Width)
    """
    n, h, w = src.shape
    full_win_W = np.ceil(np.sqrt(n))
    full_win_h = int(h * full_win_W)
    full_win_w = int(w * full_win_W)
    out = np.zeros([full_win_h, full_win_w])
    for ch in xrange(n):
        Y = ch // full_win_W
        X = ch % full_win_W
        out[Y*h:Y*h+h, X*w:X*w+w] = src[ch, :, :]
    return out


if __name__ == '__main__':
    # Init:
    if mode == 'GPU':
        caffe.set_device(0)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    net = caffe.Net(model_def, model_weights, caffe.TEST)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

    # Load mean file:
    # blob = caffe.proto.caffe_pb2.BlobProto()
    # data = open(model_mean, 'rb').read()
    # blob.ParseFromString(data)
    # mu = np.array(caffe.io.blobproto_to_array(blob))
    # # mu = mu[0, ...] / 255.0
    # mu = mu[0, ...]

    # transformer.set_transpose('data', (2, 0, 1))
    # transformer.set_transpose('data', (1, 0))
    # transformer.set_mean('data', mu)
    # transformer.set_raw_scale('data', 1/255.0)
    # transformer.set_channel_swap('data', (2, 1, 0))


    if using_image == None:
        cam = cv2.VideoCapture(cam_id)

    last_score = None
    print 'Initialization complete.'

    # Main Loop:
    for img in ['1.png', '2.png', '3.png', '4.png', '5.png', '6.png', '7.png', '8.png', '9.png']:
    # while True:
        using_image = 'test/'+img
        start = time.time()
        if using_image == None:
            # start = time.time()
            _, frame = cam.read()
        else:
            frame = cv2.imread(using_image)
            # frame = imresize(frame, [image_h, image_w])
        # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame_gray = cv2.equalizeHist(frame_gray)

        # Currently this code can only process one face, so this loop actually
        # iterate only one time:

        # input_img = frame[:, 80:560, :]
        # cv2.imshow('crop', input_img)
        
        if hist_equ:
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2YUV)
            input_img[:, :, 0] = cv2.equalizeHist(input_img[:, :, 0])
            input_img = cv2.cvtColor(input_img, cv2.COLOR_YUV2BGR)
        if blur:
            input_img = cv2.GaussianBlur(input_img, kernel, 0)
            # input_img = cv2.blur(input_img, kernel)
        if show_img_patch:
            cv2.imshow('Enhanced', input_img)

        # input_img = input_img[..., ::-1] # convert BGR to RGB
        # input_img = np.transpose(input_img, [2, 0, 1])
        # input_img = input_img[np.newaxis, ...]
        # input_img = input_img.astype(np.float32) / 255.0

        input_img = frame
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_img = input_img.astype(np.float32) / 255.0
        input_img = imresize(input_img, [image_h, image_w])
        net.blobs['data'].data[...] = transformer.preprocess('data', input_img)
        # input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # net.blobs['data'].data[...] = input_img
        output = net.forward()

        if show_middle_layers:
            conv1_out = merge_map(net.blobs['conv1'].data[0, ...])
            conv2_out = merge_map(net.blobs['conv2'].data[0, ...])
            conv3_out = merge_map(net.blobs['conv3'].data[0, ...])
            ip1_out = merge_map(net.blobs['ip1'].data[0, ...][..., np.newaxis, np.newaxis] * np.ones([7, 50, 50]))
            conv1_out /= 255.0
            cv2.imshow('conv1', conv1_out)
            cv2.imshow('conv2', conv2_out)
            cv2.imshow('conv3', conv3_out)
            cv2.imshow('ip1', ip1_out)


        prob = output['prob'][0] # the prob for the first image (the only image)

        # I am goint to implement the score decay, which can make the
        # prediction more stable:
        if last_score == None:
            pass

        prediction = np.argmax(prob)
        duration = time.time() - start
        print '%d in %fs'%(prediction, duration)
        # if prediction > 6:
            # print prediction
            # break

        # text = str(prediction) + ', prob = ' + str(prob[prediction])
        # cv2.putText(frame, text, (200, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                # thickness = 2)

        if using_image == None:
            duration = time.time() - start
            fps = 1 / duration
            text = 'fps: ' + str(fps)
            cv2.putText(frame, text, (400, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 255)
            cv2.imshow('frame', frame)
        # else:
            # pass
            # # cv2.imwrite('image_out.jpg', frame)
            # break

        # cv2.imshow('frame', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
            # break
    if using_image == None:
        cam.release()
    cv2.destroyAllWindows()
