# Caffe Digit Recognition

## Introduction
This project is designed to run Convolutional Neural Nets for handwriting digit
recognition on embedded devices like DJI Manifold and Nvidia Jetson TK1.
We are using Caffe because we found Caffe works on DJI Manifold.

## Usage
First you need to have Caffe setup correctly, and the following environment
variables are set:
```
export CAFFE_ROOT="Your directory to Caffe"
export PYTHONPATH="$CAFFE_ROOT/distribute/python:$PYTHONPATH"
export LD_LIBRARY_PATH="$CAFFE_ROOT/distribute/lib:$LD_LIBRARY_PATH"
export PATH="$CAFFE_ROOT/distribute/bin:$PATH"
```

We have already included the weights files, so you don't need to re-train the
network. Just run the following command to see the result:
```
python cam_detection.py
```

In case you want to start training, do the following steps:
```
./init.sh
./train_lenet.sh
```
