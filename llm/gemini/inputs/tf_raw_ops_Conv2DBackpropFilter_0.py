
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
# Assume generated_inputs dictionary is pre-initialized
# generated_inputs = {}

def tf_raw_ops_conv2dbackpropfilter_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.Conv2DBackpropFilter.
    """
    list_of_inputs = []

    # Input 1: Basic case with 'VALID' padding and 'NHWC' format.
    in_channels_1 = 3
    out_channels_1 = 5
    input_shape_1 = [2, 10, 10, in_channels_1]
    filter_sizes_1 = [3, 3, in_channels_1, out_channels_1]
    out_backprop_shape_1 = [2, 8, 8, out_channels_1]
    input_dict_1 = {
        'input': np.random.rand(*input_shape_1).astype(np.float32),
        'filter_sizes': np.array(filter_sizes_1, dtype=np.int32),
        'out_backprop': np.random.rand(*out_backprop_shape_1).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'valid_nhwc'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Basic case with 'SAME' padding and 'NHWC' format.
    in_channels_2 = 4
    out_channels_2 = 8
    input_shape_2 = [1, 7, 7, in_channels_2]
    filter_sizes_2 = [3, 3, in_channels_2, out_channels_2]
    out_backprop_shape_2 = [1, 7, 7, out_channels_2]
    input_dict_2 = {
        'input': np.random.rand(*input_shape_2).astype(np.float32),
        'filter_sizes': np.array(filter_sizes_2, dtype=np.int32),
        'out_backprop': np.random.rand(*out_backprop_shape_2).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'same_nhwc'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 'NCHW' data format.
    in_channels_3 = 3
    out_channels_3 = 6
    input_shape_3 = [2, in_channels_3, 12, 12]
    filter_sizes_3 = [4, 4, in_channels_3, out_channels_3]
    out_backprop_shape_3 = [2, out_channels_3, 9, 9]
    input_dict_3 = {
        'input': np.random.rand(*input_shape_3).astype(np.float32),
        'filter_sizes': np.array(filter_sizes_3, dtype=np.int32),
        'out_backprop': np.random.rand(*out_backprop_shape_3).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'use_cudnn_on_gpu': False,
        'explicit_paddings': [],
        'data_format': 'NCHW',
        'dilations': [1, 1, 1, 1],
        'name': 'valid_nchw'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Strides > 1.
    in_channels_4 = 2
    out_channels_4 = 4
    input_shape_4 = [1, 10, 10, in_channels_4]
    filter_sizes_4 = [3, 3, in_channels_4, out_channels_4]
    out_backprop_shape_4 = [1, 4, 4, out_channels_4]
    input_dict_4 = {
        'input': np.random.rand(*input_shape_4).astype(np.float32),
        'filter_sizes': np.array(filter_sizes_4, dtype=np.int32),
        'out_backprop': np.random.rand(*out_backprop_shape_4).astype(np.float32),
        'strides': [1, 2, 2, 1],
        'padding': 'VALID',
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'strided_valid_nhwc'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Dilations > 1.
    in_channels_5 = 3
    out_channels_5 = 5
    input_shape_5 = [1, 10, 10, in_channels_5]
    filter_sizes_5 = [3, 3, in_channels_5, out_channels_5]
    out_backprop_shape_5 = [1, 6, 6, out_channels_5]
    input_dict_5 = {
        'input': np.random.rand(*input_shape_5).astype(np.float32),
        'filter_sizes': np.array(filter_sizes_5, dtype=np.int32),
        'out_backprop': np.random.rand(*out_backprop_shape_5).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 2, 2, 1],
        'name': 'dilated_valid_nhwc'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: float64 dtype.
    in_channels_6 = 4
    out_channels_6 = 6
    input_shape_6 = [1, 8, 8, in_channels_6]
    filter_sizes_6 = [2, 2, in_channels_6, out_channels_6]
    out_backprop_shape_6 = [1, 4, 4, out_channels_6]
    input_dict_6 = {
        'input': np.random.rand(*input_shape_6).astype(np.float64),
        'filter_sizes': np.array(filter_sizes_6, dtype=np.int32),
        'out_backprop': np.random.rand(*out_backprop_shape_6).astype(np.float64),
        'strides': [1, 2, 2, 1],
        'padding': 'SAME',
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'float64_same_strided'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: float16 (half) dtype.
    in_channels_7 = 2
    out_channels_7 = 8
    input_shape_7 = [3, 12, 12, in_channels_7]
    filter_sizes_7 = [4, 4, in_channels_7, out_channels_7]
    out_backprop_shape_7 = [3, 9, 9, out_channels_7]
    input_dict_7 = {
        'input': np.random.rand(*input_shape_7).astype(np.float16),
        'filter_sizes': np.array(filter_sizes_7, dtype=np.int32),
        'out_backprop': np.random.rand(*out_backprop_shape_7).astype(np.float16),
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'float16_valid'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: 'EXPLICIT' padding with 'NHWC'.
    in_channels_8 = 1
    out_channels_8 = 1
    input_shape_8 = [1, 5, 5, in_channels_8]
    filter_sizes_8 = [3, 3, in_channels_8, out_channels_8]
    out_backprop_shape_8 = [1, 5, 7, out_channels_8]
    input_dict_8 = {
        'input': np.random.rand(*input_shape_8).astype(np.float32),
        'filter_sizes': np.array(filter_sizes_8, dtype=np.int32),
        'out_backprop': np.random.rand(*out_backprop_shape_8).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'EXPLICIT',
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [0, 0, 1, 1, 2, 2, 0, 0],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'explicit_nhwc'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Complex case: 'NCHW', strided, dilated, 'SAME' padding.
    in_channels_9 = 4
    out_channels_9 = 8
    input_shape_9 = [2, in_channels_9, 15, 15]
    filter_sizes_9 = [4, 4, in_channels_9, out_channels_9]
    out_backprop_shape_9 = [2, out_channels_9, 8, 8]
    input_dict_9 = {
        'input': np.random.rand(*input_shape_9).astype(np.float32),
        'filter_sizes': np.array(filter_sizes_9, dtype=np.int32),
        'out_backprop': np.random.rand(*out_backprop_shape_9).astype(np.float32),
        'strides': [1, 1, 2, 2],
        'padding': 'SAME',
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [],
        'data_format': 'NCHW',
        'dilations': [1, 1, 2, 2],
        'name': 'complex_nchw'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Minimal case (1x1 filter, no padding change).
    in_channels_11 = 1
    out_channels_11 = 1
    input_shape_11 = [1, 5, 5, in_channels_11]
    filter_sizes_11 = [1, 1, in_channels_11, out_channels_11]
    out_backprop_shape_11 = [1, 5, 5, out_channels_11]
    input_dict_11 = {
        'input': np.random.rand(*input_shape_11).astype(np.float32),
        'filter_sizes': np.array(filter_sizes_11, dtype=np.int32),
        'out_backprop': np.random.rand(*out_backprop_shape_11).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'minimal_1x1_filter'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    # Input 11: Larger dimensions, 'SAME' padding.
    in_channels_12 = 32
    out_channels_12 = 64
    input_shape_12 = [16, 32, 32, in_channels_12]
    filter_sizes_12 = [3, 3, in_channels_12, out_channels_12]
    out_backprop_shape_12 = [16, 32, 32, out_channels_12]
    input_dict_12 = {
        'input': np.random.rand(*input_shape_12).astype(np.float32),
        'filter_sizes': np.array(filter_sizes_12, dtype=np.int32),
        'out_backprop': np.random.rand(*out_backprop_shape_12).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'large_dims_same'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_12))

    return list_of_inputs

generated_inputs["tf.raw_ops.Conv2DBackpropFilter"] = tf_raw_ops_conv2dbackpropfilter_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Conv2DBackpropFilter' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Conv2DBackpropFilter'.")

check_valid('tf.raw_ops.Conv2DBackpropFilter', generated_inputs['tf.raw_ops.Conv2DBackpropFilter'], lib="tf", suffix=0)
