
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_DepthwiseConv2dNativeBackpropFilter_inputs():
    list_of_inputs = []

    # Case 1: Standard valid case with float32, NHWC, VALID padding
    input_dict_1 = {
        'strides': [1, 1, 1, 1],
        'padding': "VALID",
        'explicit_paddings': [],
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "conv_case_1",
        'input': np.random.randn(1, 3, 3, 2).astype(np.float32),
        'filter_sizes': np.array([2, 2, 2, 1], dtype=np.int32),
        'out_backprop': np.random.randn(1, 2, 2, 2).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Case 2: SAME padding with depth multiplier of 2
    input_dict_2 = {
        'strides': [1, 1, 1, 1],
        'padding': "SAME",
        'explicit_paddings': [],
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "conv_case_2",
        'input': np.random.randn(2, 4, 4, 3).astype(np.float32),
        'filter_sizes': np.array([3, 3, 3, 2], dtype=np.int32),
        'out_backprop': np.random.randn(2, 4, 4, 6).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Case 3: VALID padding with strides of 2
    input_dict_3 = {
        'strides': [1, 2, 2, 1],
        'padding': "VALID",
        'explicit_paddings': [],
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "conv_case_3",
        'input': np.random.randn(1, 5, 5, 1).astype(np.float32),
        'filter_sizes': np.array([3, 3, 1, 3], dtype=np.int32),
        'out_backprop': np.random.randn(1, 2, 2, 3).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Case 4: Larger spatial dimensions, SAME padding
    input_dict_4 = {
        'strides': [1, 2, 2, 1],
        'padding': "SAME",
        'explicit_paddings': [],
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "conv_case_4",
        'input': np.random.randn(1, 8, 8, 4).astype(np.float32),
        'filter_sizes': np.array([3, 3, 4, 1], dtype=np.int32),
        'out_backprop': np.random.randn(1, 4, 4, 4).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Case 5: Depth multiplier of 1, larger filter size, VALID padding
    input_dict_5 = {
        'strides': [1, 1, 1, 1],
        'padding': "VALID",
        'explicit_paddings': [],
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "conv_case_5",
        'input': np.random.randn(4, 16, 16, 3).astype(np.float32),
        'filter_sizes': np.array([5, 5, 3, 1], dtype=np.int32),
        'out_backprop': np.random.randn(4, 12, 12, 3).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Case 6: Depth multiplier of 2, SAME padding
    input_dict_6 = {
        'strides': [1, 1, 1, 1],
        'padding': "SAME",
        'explicit_paddings': [],
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "conv_case_6",
        'input': np.random.randn(2, 10, 10, 2).astype(np.float32),
        'filter_sizes': np.array([3, 3, 2, 2], dtype=np.int32),
        'out_backprop': np.random.randn(2, 10, 10, 4).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Case 7: Strided SAME padding with single channel
    input_dict_7 = {
        'strides': [1, 2, 2, 1],
        'padding': "SAME",
        'explicit_paddings': [],
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "conv_case_7",
        'input': np.random.randn(1, 4, 4, 1).astype(np.float32),
        'filter_sizes': np.array([2, 2, 1, 2], dtype=np.int32),
        'out_backprop': np.random.randn(1, 2, 2, 2).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Case 8: float64 dtype support
    input_dict_8 = {
        'strides': [1, 1, 1, 1],
        'padding': "VALID",
        'explicit_paddings': [],
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "conv_case_8",
        'input': np.random.randn(1, 3, 3, 2).astype(np.float64),
        'filter_sizes': np.array([2, 2, 2, 1], dtype=np.int32),
        'out_backprop': np.random.randn(1, 2, 2, 2).astype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Case 9: Uniform negative inputs
    input_dict_9 = {
        'strides': [1, 1, 1, 1],
        'padding': "VALID",
        'explicit_paddings': [],
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "conv_case_9",
        'input': np.random.uniform(-1.0, 1.0, (2, 3, 3, 2)).astype(np.float32),
        'filter_sizes': np.array([2, 2, 2, 2], dtype=np.int32),
        'out_backprop': np.random.uniform(-1.0, 1.0, (2, 2, 2, 4)).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Case 10: Strided SAME padding with depthwise_multiplier of 2
    input_dict_10 = {
        'strides': [1, 2, 2, 1],
        'padding': "SAME",
        'explicit_paddings': [],
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "conv_case_10",
        'input': np.random.randn(1, 6, 6, 2).astype(np.float32),
        'filter_sizes': np.array([3, 3, 2, 2], dtype=np.int32),
        'out_backprop': np.random.randn(1, 3, 3, 4).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.DepthwiseConv2dNativeBackpropFilter"] = tf_raw_ops_DepthwiseConv2dNativeBackpropFilter_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.DepthwiseConv2dNativeBackpropFilter' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DepthwiseConv2dNativeBackpropFilter'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.DepthwiseConv2dNativeBackpropFilter', generated_inputs['tf.raw_ops.DepthwiseConv2dNativeBackpropFilter'], lib="tf", suffix=0)
