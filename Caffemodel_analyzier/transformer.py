#!/usr/bin/env python

"""transformer.py: Input data need to be reshaped and transform to the type that caffe could accept"""

__author__ = "HaojieYuan"
__email__ = "haojie.d.yuan@gmail.com"
__version__ = "0.1"


import sys
import config
import numpy as np
sys.path.append(config.pycaffe_path)
import caffe


def set(net, Mean_file, bin2npy):
    """Set transformer for image input,
    net should be created by caffe.Net() and Mean_file can be binarypro and npy.
    And if it's binaryproto, remember to set bin2npy to True"""
    
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255.0)

    if bin2npy == True:
        blob = caffe.proto.caffe_pb2.BlobProto()
        data = open( Mean_file , 'rb' ).read()
        blob.ParseFromString(data)
        arr = np.array( caffe.io.blobproto_to_array(blob) )
        out = arr[0]
        np.save( Mean_file + '2npy.npy' , out )
        Mean_file = Mean_file + '2npy.npy'

    transformer.set_mean('data', np.load(Mean_file).mean(1).mean(1))

    return transformer
