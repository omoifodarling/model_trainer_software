from __future__ import print_function # Use a function definition from future version (say 3.x from 2.7 interpreter)
from IPython.display import Image
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import PIL
from CNN import create_reader,evaluate_model,train_and_evaluate,save_model,myimg
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

# Create the train and test readers
reader_train = create_reader(os.path.join(data_path, 'train_map.txt'), 
                             os.path.join(data_path, 'CIFAR-10_mean.xml'), True)
reader_test  = create_reader(os.path.join(data_path, 'test_map.txt'), 
                             os.path.join(data_path, 'CIFAR-10_mean.xml'), False)


def create_basic_model_with_dropout(input, out_dims):

    with C.layers.default_options(activation=C.relu, init=C.glorot_uniform()):
        model = C.layers.Sequential([
            C.layers.For(range(3), lambda i: [
                C.layers.Convolution((5,5), [32,32,64][i], pad=True),
                C.layers.MaxPooling((3,3), strides=(2,2))
            ]),
            C.layers.Dense(64),
            C.layers.Dropout(0.25),
            C.layers.Dense(out_dims, activation=None)
        ])
    print('Basic Model Prediction WITH DROPOUT')
	
    return model(input)
		
pred_basic_model_dropout = train_and_evaluate(reader_train, 
                                              reader_test, 
                                              max_epochs=50, 
                                              model_func=create_basic_model_with_dropout,parX=0.01,parY=0.003,parZ=0.001,size_per_iter=250)
	
evaluate_model(pred_basic_model_dropout,myimg)

save_model('pred_model_dropout_50.dnn',pred_basic_model_dropout)

