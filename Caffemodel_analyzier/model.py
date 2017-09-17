#!/usr/bin/env python

"""model.py: Read a pre trained caffe model"""

__author__ = "HaojieYuan"
__email__ = "haojie.d.yuan@gmail.com"
__version__ = "0.1"


import sys
import config
sys.path.append(config.pycaffe_path)
import caffe

def read(use_GPU, GPU_device, LAYER_FILE, PRETRAINED,
               image_batch, reshape_channel, reshape_height, reshape_width):
    """Return a caffe.net class object with given model.
    Running mode can be set to GPU by adjusting config.
    But CPU is recommanded as this is just testing a pre-trained model, not training."""

    # Set mode CPU or GPU
    if use_GPU == True:
        caffe.set_mode_gpu()
        caffe.set_device(GPU_device)
    else:
        caffe.set_mode_cpu()

    # Load pretrained network weights from snapshot.
    net = caffe.Net(LAYER_FILE, PRETRAINED, caffe.TEST)
    net.blobs['data'].reshape(image_batch, reshape_channel, reshape_height, reshape_width)

    return net
