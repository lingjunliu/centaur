
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_raw_ops_Conv2DBackpropInput_inputs():
    list_of_inputs = []
    np.random.seed(42)

    # Input 1: NHWC, VALID, float32
    input_dict = {
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_1',
        'input_sizes': np.array([2, 5, 5, 3], dtype=np.int32),
        'filter': np.random.randn(3, 3, 3, 8).astype(np.float32),
        'out_backprop': np.random.randn(2, 3, 3, 8).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: NHWC, SAME, stride 2, float32
    input_dict = {
        'strides': [1, 2, 2, 1],
        'padding': 'SAME',
        'use_cudnn_on_gpu': False,
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_2',
        'input_sizes': np.array([2, 6, 6, 3], dtype=np.int32),
        'filter': np.random.randn(3, 3, 3, 8).astype(np.float32),
        'out_backprop': np.random.randn(2, 3, 3, 8).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: NHWC, VALID, float32 (originally NCHW, converted to NHWC)
    input_dict = {
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_3',
        'input_sizes': np.array([2, 5, 5, 3], dtype=np.int32),
        'filter': np.random.randn(3, 3, 3, 8).astype(np.float32),
        'out_backprop': np.random.randn(2, 3, 3, 8).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: NHWC, EXPLICIT, float32
    input_dict = {
        'strides': [1, 1, 1, 1],
        'padding': 'EXPLICIT',
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [0, 0, 1, 2, 3, 4, 0, 0],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_4',
        'input_sizes': np.array([2, 5, 5, 3], dtype=np.int32),
        'filter': np.random.randn(3, 3, 3, 8).astype(np.float32),
        'out_backprop': np.random.randn(2, 6, 10, 8).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: NHWC, dilation = 2, VALID, float32
    input_dict = {
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 2, 2, 1],
        'name': 'conv_5',
        'input_sizes': np.array([2, 5, 5, 3], dtype=np.int32),
        'filter': np.random.randn(3, 3, 3, 8).astype(np.float32),
        'out_backprop': np.random.randn(2, 1, 1, 8).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float64, NHWC, VALID
    input_dict = {
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'use_cudnn_on_gpu': False,
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_6',
        'input_sizes': np.array([1, 4, 4, 1], dtype=np.int32),
        'filter': np.random.randn(2, 2, 1, 1).astype(np.float64),
        'out_backprop': np.random.randn(1, 3, 3, 1).astype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: NHWC, SAME, stride 2, float32 (originally NCHW, converted to NHWC)
    input_dict = {
        'strides': [1, 2, 2, 1],
        'padding': 'SAME',
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_7',
        'input_sizes': np.array([2, 8, 8, 4], dtype=np.int32),
        'filter': np.random.randn(3, 3, 4, 16).astype(np.float32),
        'out_backprop': np.random.randn(2, 4, 4, 16).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: NHWC, EXPLICIT, float32 (originally NCHW, converted to NHWC)
    input_dict = {
        'strides': [1, 1, 1, 1],
        'padding': 'EXPLICIT',
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [0, 0, 1, 1, 2, 2, 0, 0],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_8',
        'input_sizes': np.array([2, 6, 6, 2], dtype=np.int32),
        'filter': np.random.randn(3, 3, 2, 4).astype(np.float32),
        'out_backprop': np.random.randn(2, 6, 8, 4).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: NHWC, dilation with strides, float32
    input_dict = {
        'strides': [1, 2, 2, 1],
        'padding': 'VALID',
        'use_cudnn_on_gpu': False,
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 2, 2, 1],
        'name': 'conv_9',
        'input_sizes': np.array([4, 7, 7, 3], dtype=np.int32),
        'filter': np.random.randn(3, 3, 3, 2).astype(np.float32),
        'out_backprop': np.random.randn(4, 2, 2, 2).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32, larger channels, NHWC, SAME
    input_dict = {
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_10',
        'input_sizes': np.array([1, 14, 14, 64], dtype=np.int32),
        'filter': np.random.randn(3, 3, 64, 128).astype(np.float32),
        'out_backprop': np.random.randn(1, 14, 14, 128).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Conv2DBackpropInput"] = tf_raw_ops_Conv2DBackpropInput_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Conv2DBackpropInput' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Conv2DBackpropInput'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Conv2DBackpropInput', generated_inputs['tf.raw_ops.Conv2DBackpropInput'], lib="tf", suffix=0)
