
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_nn_depthwise_conv2d_backprop_input_inputs():
    list_of_inputs = []
    
    # 1. NHWC, SAME, stride 1, float32
    input_sizes = np.array([2, 8, 8, 3], dtype=np.int32)
    filter_val = np.random.randn(3, 3, 3, 2).astype(np.float32)
    out_backprop = np.random.randn(2, 8, 8, 6).astype(np.float32)
    input_dict = {
        'input_sizes': input_sizes,
        'filter': filter_val,
        'out_backprop': out_backprop,
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'test1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2. NHWC, VALID, stride 1, float32
    input_sizes = np.array([1, 5, 5, 2], dtype=np.int32)
    filter_val = np.random.randn(3, 3, 2, 1).astype(np.float32)
    out_backprop = np.random.randn(1, 3, 3, 2).astype(np.float32)
    input_dict = {
        'input_sizes': input_sizes,
        'filter': filter_val,
        'out_backprop': out_backprop,
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'test2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3. NHWC, SAME, stride 2, float32
    input_sizes = np.array([2, 6, 6, 4], dtype=np.int32)
    filter_val = np.random.randn(3, 3, 4, 2).astype(np.float32)
    out_backprop = np.random.randn(2, 3, 3, 8).astype(np.float32)
    input_dict = {
        'input_sizes': input_sizes,
        'filter': filter_val,
        'out_backprop': out_backprop,
        'strides': [1, 2, 2, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'test3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 4. NHWC, VALID, stride 1, float32, smaller size
    input_sizes = np.array([1, 4, 4, 1], dtype=np.int32)
    filter_val = np.random.randn(2, 2, 1, 1).astype(np.float32)
    out_backprop = np.random.randn(1, 3, 3, 1).astype(np.float32)
    input_dict = {
        'input_sizes': input_sizes,
        'filter': filter_val,
        'out_backprop': out_backprop,
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'test4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 5. NHWC, VALID, stride 2, float32
    input_sizes = np.array([1, 6, 6, 3], dtype=np.int32)
    filter_val = np.random.randn(3, 3, 3, 1).astype(np.float32)
    out_backprop = np.random.randn(1, 2, 2, 3).astype(np.float32)
    input_dict = {
        'input_sizes': input_sizes,
        'filter': filter_val,
        'out_backprop': out_backprop,
        'strides': [1, 2, 2, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'test5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 6. NHWC, SAME, stride 1, multiplier=2
    input_sizes = np.array([2, 4, 4, 2], dtype=np.int32)
    filter_val = np.random.randn(2, 2, 2, 2).astype(np.float32)
    out_backprop = np.random.randn(2, 4, 4, 4).astype(np.float32)
    input_dict = {
        'input_sizes': input_sizes,
        'filter': filter_val,
        'out_backprop': out_backprop,
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'test6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 7. NHWC, VALID, stride 2, multiplier=3
    input_sizes = np.array([1, 7, 7, 1], dtype=np.int32)
    filter_val = np.random.randn(3, 3, 1, 3).astype(np.float32)
    out_backprop = np.random.randn(1, 3, 3, 3).astype(np.float32)
    input_dict = {
        'input_sizes': input_sizes,
        'filter': filter_val,
        'out_backprop': out_backprop,
        'strides': [1, 2, 2, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'test7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 8. float64 type, NHWC, SAME, stride 1
    input_sizes = np.array([2, 4, 4, 1], dtype=np.int32)
    filter_val = np.random.randn(2, 2, 1, 1).astype(np.float64)
    out_backprop = np.random.randn(2, 4, 4, 1).astype(np.float64)
    input_dict = {
        'input_sizes': input_sizes,
        'filter': filter_val,
        'out_backprop': out_backprop,
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'test8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 9. float16 type, NHWC, VALID, stride 1
    input_sizes = np.array([1, 3, 3, 2], dtype=np.int32)
    filter_val = np.random.randn(2, 2, 2, 2).astype(np.float16)
    out_backprop = np.random.randn(1, 2, 2, 4).astype(np.float16)
    input_dict = {
        'input_sizes': input_sizes,
        'filter': filter_val,
        'out_backprop': out_backprop,
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'test9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 10. float32 with negative values, NHWC, VALID
    input_sizes = np.array([1, 4, 4, 1], dtype=np.int32)
    filter_val = np.array([[[[-1.0]], [[2.0]]], [[[0.5]], [[-1.5]]]], dtype=np.float32)
    out_backprop = np.random.randn(1, 3, 3, 1).astype(np.float32)
    input_dict = {
        'input_sizes': input_sizes,
        'filter': filter_val,
        'out_backprop': out_backprop,
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'test10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.depthwise_conv2d_backprop_input"] = tf_nn_depthwise_conv2d_backprop_input_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.depthwise_conv2d_backprop_input' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.depthwise_conv2d_backprop_input'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.depthwise_conv2d_backprop_input', generated_inputs['tf.nn.depthwise_conv2d_backprop_input'], lib="tf", suffix=0)
