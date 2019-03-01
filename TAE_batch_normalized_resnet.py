from __future__ import print_function # Use a function definition from future version (say 3.x from 2.7 interpreter)
from IPython.display import Image
import matplotlib.pyplot as plt
import math
import numpy as np
import os
from CNN import create_reader,evaluate_model,train_and_evaluate,save_model,myimg
import PIL
import sys


try:
    from urllib.request import urlopen
except ImportError:
    from urllib import urlopen


import cntk as C

if 'TEST_DEVICE' in os.environ:
    if os.environ['TEST_DEVICE'] == 'cpu':
        C.device.try_set_default_device(C.device.cpu())
    else:
        C.device.try_set_default_device(C.device.gpu(0))
        
data_path = os.path.join('data', 'CIFAR-10')

# model dimensions
image_height = 32
image_width  = 32
num_channels = 3
num_classes  = 10

import cntk.io.transforms as xforms 


def convolution_bn(input, filter_size, num_filters, strides=(1,1), init=C.he_normal(), activation=C.relu):
    if activation is None:
        activation = lambda x: x
        
    r = C.layers.Convolution(filter_size, 
                             num_filters, 
                             strides=strides, 
                             init=init, 
                             activation=None, 
                             pad=True, bias=False)(input)
    r = C.layers.BatchNormalization(map_rank=1)(r)
    r = activation(r)
    
    return r

def resnet_basic(input, num_filters):
    c1 = convolution_bn(input, (3,3), num_filters)
    c2 = convolution_bn(c1, (3,3), num_filters, activation=None)
    p  = c2 + input
    return C.relu(p)

def resnet_basic_inc(input, num_filters):
    c1 = convolution_bn(input, (3,3), num_filters, strides=(2,2))
    c2 = convolution_bn(c1, (3,3), num_filters, activation=None)

    s = convolution_bn(input, (1,1), num_filters, strides=(2,2), activation=None)
    
    p = c2 + s
    return C.relu(p)

def resnet_basic_stack(input, num_filters, num_stack):
    assert (num_stack > 0)
    
    r = input
    for _ in range(num_stack):
        r = resnet_basic(r, num_filters)
    return r
                                                        
def create_resnet_model(input, out_dims):
    conv = convolution_bn(input, (3,3), 16)
    r1_1 = resnet_basic_stack(conv, 16, 3)

    r2_1 = resnet_basic_inc(r1_1, 32)
    r2_2 = resnet_basic_stack(r2_1, 32, 2)

    r3_1 = resnet_basic_inc(r2_2, 64)
    r3_2 = resnet_basic_stack(r3_1, 64, 2)

    # Global average pooling
    pool = C.layers.AveragePooling(filter_shape=(8,8), strides=(1,1))(r3_2)    
    net = C.layers.Dense(out_dims, init=C.he_normal(), activation=None)(pool)
    #print('Basic STACK Prediction WITH RESNET_MODEL')
    return net    
    
pred_resnet = train_and_evaluate(reader_train, reader_test, max_epochs=45, model_func=create_resnet_model,parX=0.02,parY=0.005,parZ=0.0018,size_per_iter=250)
evaluate_model(pred_resnet,myimg)

save_model('pred_model_resnet.dnn',pred_resnet)
