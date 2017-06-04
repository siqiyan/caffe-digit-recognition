#!/bin/bash

# This script downloads required dataset on the right place, apply data
# augmentation, and convert the dataset into LMDB format, which is required by
# Caffe.
# In order to run this script, you need to make sure that Caffe is setup
# correctly and PATH include $CAFFE/distribute/bin

ROOT=dataset
TRAIN_DIR=mnist_aug_train
TEST_DIR=mnist_aug_test
LMDB_EXEC=convert_imageset.bin
original=$PWD

mkdir MNIST
cd MNIST
if [ ! -f train-images-idx3-ubyte ]; then
    wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    gzip -d train-images-idx3-ubyte.gz
fi
if [ ! -f train-labels-idx1-ubyte ]; then
    wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
    gzip -d train-labels-idx1-ubyte.gz
fi
if [ ! -f t10k-images-idx3-ubyte ]; then
    wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
    gzip -d t10k-images-idx3-ubyte.gz
fi
if [ ! -f t10k-labels-idx1-ubyte ]; then
    wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
    gzip -d t10k-labels-idx1-ubyte.gz
fi

cd $original
if [ ! -d $ROOT ]; then
    mkdir $ROOT
fi

cd $ROOT
rm -rf "$TRAIN_DIR"
rm -rf "$TEST_DIR"
rm -f "$TRAIN_DIR.list"
rm -f "$TEST_DIR.list"
rm -rf lmdb*
mkdir "$TRAIN_DIR"
mkdir "$TEST_DIR"
cd $original
mkdir weights

python convert_mnist.py
$LMDB_EXEC --gray $ROOT/ $ROOT/$TRAIN_DIR.list $ROOT/lmdb_$TRAIN_DIR
$LMDB_EXEC --gray $ROOT/ $ROOT/$TEST_DIR.list $ROOT/lmdb_$TEST_DIR
