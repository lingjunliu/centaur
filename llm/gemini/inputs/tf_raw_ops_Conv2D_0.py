
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Conv2D_inputs():
    list_of_inputs = []
    
    # Input 1: Standard float32 NHWC, valid padding, stride 1
    input_dict = {
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_1',
        'input': np.random.randn(2, 8, 8, 3).astype(np.float32),
        'filter': np.random.randn(3, 3, 3, 16).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'VALID'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Standard float32 NHWC, same padding, stride 2
    input_dict = {
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_2',
        'input': np.random.randn(1, 16, 16, 4).astype(np.float32),
        'filter': np.random.randn(5, 5, 4, 8).astype(np.float32),
        'strides': [1, 2, 2, 1],
        'padding': 'SAME'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float64 NHWC, same padding, stride 1, dilation 2
    input_dict = {
        'use_cudnn_on_gpu': False,
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 2, 2, 1],
        'name': 'conv_3',
        'input': np.random.randn(2, 10, 10, 3).astype(np.float64),
        'filter': np.random.randn(3, 3, 3, 4).astype(np.float64),
        'strides': [1, 1, 1, 1],
        'padding': 'SAME'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Float16 NHWC, valid padding, stride 1
    input_dict = {
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_4',
        'input': np.random.randn(1, 14, 14, 1).astype(np.float16),
        'filter': np.random.randn(3, 3, 1, 2).astype(np.float16),
        'strides': [1, 1, 1, 1],
        'padding': 'VALID'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Int32 NHWC, same padding, stride 1
    input_dict = {
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_5',
        'input': np.random.randint(-10, 10, size=(1, 6, 6, 2)).astype(np.int32),
        'filter': np.random.randint(-5, 5, size=(2, 2, 2, 3)).astype(np.int32),
        'strides': [1, 1, 1, 1],
        'padding': 'SAME'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: NCHW, Float32, same padding, stride 1
    input_dict = {
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [],
        'data_format': 'NCHW',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_6',
        'input': np.random.randn(2, 3, 12, 12).astype(np.float32),
        'filter': np.random.randn(3, 3, 3, 8).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'SAME'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: NCHW, Float32, valid padding, stride 2
    input_dict = {
        'use_cudnn_on_gpu': False,
        'explicit_paddings': [],
        'data_format': 'NCHW',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_7',
        'input': np.random.randn(4, 2, 16, 16).astype(np.float32),
        'filter': np.random.randn(3, 3, 2, 4).astype(np.float32),
        'strides': [1, 1, 2, 2],
        'padding': 'VALID'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: EXPLICIT padding NHWC, Float32
    input_dict = {
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [0, 0, 1, 1, 2, 2, 0, 0],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_8',
        'input': np.random.randn(1, 8, 8, 3).astype(np.float32),
        'filter': np.random.randn(3, 3, 3, 5).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'EXPLICIT'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: EXPLICIT padding NCHW, Float32
    input_dict = {
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [0, 0, 0, 0, 1, 1, 2, 2],
        'data_format': 'NCHW',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_9',
        'input': np.random.randn(1, 3, 8, 8).astype(np.float32),
        'filter': np.random.randn(3, 3, 3, 5).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'EXPLICIT'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: NHWC, Float32, dilation 3, same padding, stride 1
    input_dict = {
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 3, 3, 1],
        'name': 'conv_10',
        'input': np.random.randn(1, 20, 20, 3).astype(np.float32),
        'filter': np.random.randn(3, 3, 3, 4).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'SAME'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Conv2D"] = tf_raw_ops_Conv2D_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Conv2D' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Conv2D'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Conv2D', generated_inputs['tf.raw_ops.Conv2D'], lib="tf", suffix=0)
