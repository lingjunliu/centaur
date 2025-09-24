
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_depthwiseconv2dnativebackpropfilter_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.DepthwiseConv2dNativeBackpropFilter.
    """
    list_of_inputs = []

    # Case 1: Basic NHWC, VALID padding
    input_dict_1 = {
        'input': np.random.rand(1, 5, 5, 3).astype(np.float32),
        'filter_sizes': np.array([3, 3, 3, 1], dtype=np.int32),
        'out_backprop': np.random.rand(1, 3, 3, 3).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'test_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Case 2: Basic NHWC, SAME padding
    input_dict_2 = {
        'input': np.random.rand(1, 5, 5, 3).astype(np.float32),
        'filter_sizes': np.array([3, 3, 3, 1], dtype=np.int32),
        'out_backprop': np.random.rand(1, 5, 5, 3).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'test_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Case 3: NHWC, VALID padding
    input_dict_3 = {
        'input': np.random.rand(1, 5, 5, 3).astype(np.float32),
        'filter_sizes': np.array([3, 3, 3, 1], dtype=np.int32),
        'out_backprop': np.random.rand(1, 3, 3, 3).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'test_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Case 4: Strides > 1 (NHWC, VALID)
    input_dict_4 = {
        'input': np.random.rand(1, 7, 7, 2).astype(np.float32),
        'filter_sizes': np.array([3, 3, 2, 1], dtype=np.int32),
        'out_backprop': np.random.rand(1, 3, 3, 2).astype(np.float32),
        'strides': [1, 2, 2, 1],
        'padding': 'VALID',
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'test_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Case 5: Dilations > 1 (NHWC, SAME)
    input_dict_5 = {
        'input': np.random.rand(1, 5, 5, 4).astype(np.float32),
        'filter_sizes': np.array([3, 3, 4, 1], dtype=np.int32),
        'out_backprop': np.random.rand(1, 5, 5, 4).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 2, 2, 1],
        'name': 'test_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Case 6: float64 data type, batch_size > 1
    input_dict_6 = {
        'input': np.random.rand(2, 4, 4, 1).astype(np.float64),
        'filter_sizes': np.array([2, 2, 1, 1], dtype=np.int32),
        'out_backprop': np.random.rand(2, 3, 3, 1).astype(np.float64),
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'test_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Case 7: float16 data type (NHWC)
    input_dict_7 = {
        'input': np.random.rand(1, 8, 8, 3).astype(np.float16),
        'filter_sizes': np.array([3, 3, 3, 1], dtype=np.int32),
        'out_backprop': np.random.rand(1, 4, 4, 3).astype(np.float16),
        'strides': [1, 2, 2, 1],
        'padding': 'SAME',
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'test_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Case 8: EXPLICIT padding (NHWC)
    input_dict_8 = {
        'input': np.random.rand(1, 5, 5, 2).astype(np.float32),
        'filter_sizes': np.array([3, 3, 2, 1], dtype=np.int32),
        'out_backprop': np.random.rand(1, 5, 7, 2).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'EXPLICIT',
        'explicit_paddings': [0, 0, 1, 1, 2, 2, 0, 0],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'test_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Case 9: Depthwise Multiplier > 1 (NHWC, VALID)
    input_dict_9 = {
        'input': np.random.rand(1, 6, 6, 2).astype(np.float32),
        'filter_sizes': np.array([3, 3, 2, 3], dtype=np.int32),
        'out_backprop': np.random.rand(1, 4, 4, 6).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'test_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Case 10: Non-square filter and image (NHWC, VALID)
    input_dict_10 = {
        'input': np.random.rand(1, 5, 7, 3).astype(np.float32),
        'filter_sizes': np.array([2, 4, 3, 1], dtype=np.int32),
        'out_backprop': np.random.rand(1, 4, 4, 3).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'test_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.DepthwiseConv2dNativeBackpropFilter"] = get_depthwiseconv2dnativebackpropfilter_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.DepthwiseConv2dNativeBackpropFilter' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DepthwiseConv2dNativeBackpropFilter'.")

check_valid('tf.raw_ops.DepthwiseConv2dNativeBackpropFilter', generated_inputs['tf.raw_ops.DepthwiseConv2dNativeBackpropFilter'], lib="tf", suffix=0)
