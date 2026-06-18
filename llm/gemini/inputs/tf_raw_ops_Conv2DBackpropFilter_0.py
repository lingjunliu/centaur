
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf

def tf_raw_ops_Conv2DBackpropFilter_inputs():
    list_of_inputs = []

    # Case 1: NHWC, SAME, float32, stride 1
    input_val = np.random.randn(2, 4, 4, 3).astype(np.float32)
    filter_sizes = np.array([3, 3, 3, 8], dtype=np.int32)
    out_backprop = np.random.randn(2, 4, 4, 8).astype(np.float32)
    input_dict = {
        'input': input_val,
        'filter_sizes': filter_sizes,
        'out_backprop': out_backprop,
        'strides': [1, 1, 1, 1],
        'padding': "SAME",
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [],
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "conv2d_bprop_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: NHWC, VALID, float32, stride 1
    input_val = np.random.randn(1, 5, 5, 2).astype(np.float32)
    filter_sizes = np.array([3, 3, 2, 4], dtype=np.int32)
    out_backprop = np.random.randn(1, 3, 3, 4).astype(np.float32)
    input_dict = {
        'input': input_val,
        'filter_sizes': filter_sizes,
        'out_backprop': out_backprop,
        'strides': [1, 1, 1, 1],
        'padding': "VALID",
        'use_cudnn_on_gpu': False,
        'explicit_paddings': [],
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "conv2d_bprop_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: NHWC, VALID, float32, stride 2
    input_val = np.random.randn(2, 6, 6, 3).astype(np.float32)
    filter_sizes = np.array([2, 2, 3, 2], dtype=np.int32)
    out_backprop = np.random.randn(2, 3, 3, 2).astype(np.float32)
    input_dict = {
        'input': input_val,
        'filter_sizes': filter_sizes,
        'out_backprop': out_backprop,
        'strides': [1, 2, 2, 1],
        'padding': "VALID",
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [],
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "conv2d_bprop_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: NHWC, VALID, float32, stride 1 (changed from NCHW)
    input_val = np.random.randn(2, 5, 5, 3).astype(np.float32)
    filter_sizes = np.array([3, 3, 3, 4], dtype=np.int32)
    out_backprop = np.random.randn(2, 3, 3, 4).astype(np.float32)
    input_dict = {
        'input': input_val,
        'filter_sizes': filter_sizes,
        'out_backprop': out_backprop,
        'strides': [1, 1, 1, 1],
        'padding': "VALID",
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [],
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "conv2d_bprop_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: NHWC, SAME, float32, stride 2
    input_val = np.random.randn(1, 8, 8, 4).astype(np.float32)
    filter_sizes = np.array([4, 4, 4, 2], dtype=np.int32)
    out_backprop = np.random.randn(1, 4, 4, 2).astype(np.float32)
    input_dict = {
        'input': input_val,
        'filter_sizes': filter_sizes,
        'out_backprop': out_backprop,
        'strides': [1, 2, 2, 1],
        'padding': "SAME",
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [],
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "conv2d_bprop_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: NHWC, VALID, float32, stride 1, dilation 2
    input_val = np.random.randn(1, 7, 7, 2).astype(np.float32)
    filter_sizes = np.array([3, 3, 2, 2], dtype=np.int32)
    out_backprop = np.random.randn(1, 3, 3, 2).astype(np.float32)
    input_dict = {
        'input': input_val,
        'filter_sizes': filter_sizes,
        'out_backprop': out_backprop,
        'strides': [1, 1, 1, 1],
        'padding': "VALID",
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [],
        'data_format': "NHWC",
        'dilations': [1, 2, 2, 1],
        'name': "conv2d_bprop_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: NHWC, EXPLICIT, float32, stride 1
    input_val = np.random.randn(1, 3, 3, 1).astype(np.float32)
    filter_sizes = np.array([3, 3, 1, 1], dtype=np.int32)
    out_backprop = np.random.randn(1, 3, 3, 1).astype(np.float32)
    input_dict = {
        'input': input_val,
        'filter_sizes': filter_sizes,
        'out_backprop': out_backprop,
        'strides': [1, 1, 1, 1],
        'padding': "EXPLICIT",
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [0, 0, 1, 1, 1, 1, 0, 0],
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "conv2d_bprop_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 8: NHWC, EXPLICIT, float32, stride 1 (changed from NCHW)
    input_val = np.random.randn(1, 3, 3, 1).astype(np.float32)
    filter_sizes = np.array([3, 3, 1, 1], dtype=np.int32)
    out_backprop = np.random.randn(1, 3, 3, 1).astype(np.float32)
    input_dict = {
        'input': input_val,
        'filter_sizes': filter_sizes,
        'out_backprop': out_backprop,
        'strides': [1, 1, 1, 1],
        'padding': "EXPLICIT",
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [0, 0, 1, 1, 1, 1, 0, 0],
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "conv2d_bprop_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 9: NHWC, VALID, float64, stride 1
    input_val = np.random.randn(1, 4, 4, 1).astype(np.float64)
    filter_sizes = np.array([2, 2, 1, 1], dtype=np.int32)
    out_backprop = np.random.randn(1, 3, 3, 1).astype(np.float64)
    input_dict = {
        'input': input_val,
        'filter_sizes': filter_sizes,
        'out_backprop': out_backprop,
        'strides': [1, 1, 1, 1],
        'padding': "VALID",
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [],
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "conv2d_bprop_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 10: NHWC, SAME, float16, stride 1
    input_val = np.random.randn(2, 5, 5, 3).astype(np.float16)
    filter_sizes = np.array([3, 3, 3, 4], dtype=np.int32)
    out_backprop = np.random.randn(2, 5, 5, 4).astype(np.float16)
    input_dict = {
        'input': input_val,
        'filter_sizes': filter_sizes,
        'out_backprop': out_backprop,
        'strides': [1, 1, 1, 1],
        'padding': "SAME",
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [],
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "conv2d_bprop_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Conv2DBackpropFilter"] = tf_raw_ops_Conv2DBackpropFilter_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Conv2DBackpropFilter' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Conv2DBackpropFilter'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Conv2DBackpropFilter', generated_inputs['tf.raw_ops.Conv2DBackpropFilter'], lib="tf", suffix=0)
