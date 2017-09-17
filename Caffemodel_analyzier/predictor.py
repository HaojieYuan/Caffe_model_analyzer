#!/usr/bin/env python

"""predictor.py: Make prediction with given model and imput image"""

__author__ = "HaojieYuan"
__email__ = "haojie.d.yuan@gmail.com"
__version__ = "0.1"

import numpy as np
import sys
import config
sys.path.append(config.pycaffe_path)
import caffe

def predict(image_path, transformer, net):
    """A function to make prediction with given model and input image
    image_path is path of image, absolute path is recommanded
    transformer should be presetted, which can be created by caffe.io.Transformer
    net should be created by caffe.Net"""

    input_image = caffe.io.load_image(image_path)
    net.blobs['data'].data[...] = transformer.preprocess('data', input_image)
    out = net.forward()
    result = out['prob'].argmax()

    return result
