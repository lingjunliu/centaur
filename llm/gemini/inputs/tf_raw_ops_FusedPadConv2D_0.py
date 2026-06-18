
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_FusedPadConv2D_inputs():
    list_of_inputs = []
    
    # Input 1
    input_val = np.random.randn(1, 5, 5, 3).astype(np.float32)
    paddings_val = np.array([[0, 0], [1, 1], [1, 1], [0, 0]], dtype=np.int32)
    filter_val = np.random.randn(3, 3, 3, 2).astype(np.float32)
    input_dict = {
        'mode': 'REFLECT',
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'name': 'fused_pad_conv_1',
        'input': input_val,
        'paddings': paddings_val,
        'filter': filter_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_val = np.random.randn(1, 5, 5, 3).astype(np.float32)
    paddings_val = np.array([[0, 0], [1, 1], [1, 1], [0, 0]], dtype=np.int32)
    filter_val = np.random.randn(3, 3, 3, 2).astype(np.float32)
    input_dict = {
        'mode': 'SYMMETRIC',
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'name': 'fused_pad_conv_2',
        'input': input_val,
        'paddings': paddings_val,
        'filter': filter_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_val = np.random.randn(2, 10, 10, 4).astype(np.float64)
    paddings_val = np.array([[0, 0], [2, 2], [2, 2], [0, 0]], dtype=np.int32)
    filter_val = np.random.randn(5, 5, 4, 8).astype(np.float64)
    input_dict = {
        'mode': 'REFLECT',
        'strides': [1, 2, 2, 1],
        'padding': 'VALID',
        'name': 'fused_pad_conv_3',
        'input': input_val,
        'paddings': paddings_val,
        'filter': filter_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_val = np.random.randn(1, 6, 6, 1).astype(np.float16)
    paddings_val = np.array([[0, 0], [1, 2], [1, 2], [0, 0]], dtype=np.int32)
    filter_val = np.random.randn(2, 2, 1, 2).astype(np.float16)
    input_dict = {
        'mode': 'SYMMETRIC',
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'name': 'fused_pad_conv_4',
        'input': input_val,
        'paddings': paddings_val,
        'filter': filter_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_val = np.random.randn(4, 8, 8, 2).astype(np.float32)
    paddings_val = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.int32)
    filter_val = np.random.randn(1, 1, 2, 4).astype(np.float32)
    input_dict = {
        'mode': 'REFLECT',
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'name': 'fused_pad_conv_5',
        'input': input_val,
        'paddings': paddings_val,
        'filter': filter_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_val = np.random.randn(2, 15, 15, 3).astype(np.float32)
    paddings_val = np.array([[0, 0], [3, 3], [3, 3], [0, 0]], dtype=np.int32)
    filter_val = np.random.randn(7, 7, 3, 5).astype(np.float32)
    input_dict = {
        'mode': 'SYMMETRIC',
        'strides': [1, 3, 3, 1],
        'padding': 'VALID',
        'name': 'fused_pad_conv_6',
        'input': input_val,
        'paddings': paddings_val,
        'filter': filter_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_val = np.random.randn(1, 4, 4, 8).astype(np.float16)
    paddings_val = np.array([[0, 0], [1, 1], [1, 1], [0, 0]], dtype=np.int32)
    filter_val = np.random.randn(3, 3, 8, 16).astype(np.float16)
    input_dict = {
        'mode': 'REFLECT',
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'name': 'fused_pad_conv_7',
        'input': input_val,
        'paddings': paddings_val,
        'filter': filter_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_val = np.random.randn(1, 3, 3, 1).astype(np.float64)
    paddings_val = np.array([[0, 0], [1, 1], [1, 1], [0, 0]], dtype=np.int32)
    filter_val = np.random.randn(3, 3, 1, 1).astype(np.float64)
    input_dict = {
        'mode': 'SYMMETRIC',
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'name': 'fused_pad_conv_8',
        'input': input_val,
        'paddings': paddings_val,
        'filter': filter_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_val = np.random.randn(3, 12, 12, 4).astype(np.float32)
    paddings_val = np.array([[0, 0], [2, 1], [2, 1], [0, 0]], dtype=np.int32)
    filter_val = np.random.randn(4, 4, 4, 2).astype(np.float32)
    input_dict = {
        'mode': 'REFLECT',
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'name': 'fused_pad_conv_9',
        'input': input_val,
        'paddings': paddings_val,
        'filter': filter_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_val = np.random.randn(1, 20, 20, 3).astype(np.float32)
    paddings_val = np.array([[0, 0], [4, 4], [4, 4], [0, 0]], dtype=np.int32)
    filter_val = np.random.randn(5, 5, 3, 3).astype(np.float32)
    input_dict = {
        'mode': 'SYMMETRIC',
        'strides': [1, 2, 2, 1],
        'padding': 'SAME',
        'name': 'fused_pad_conv_10',
        'input': input_val,
        'paddings': paddings_val,
        'filter': filter_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.FusedPadConv2D"] = tf_raw_ops_FusedPadConv2D_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.FusedPadConv2D' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FusedPadConv2D'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.FusedPadConv2D', generated_inputs['tf.raw_ops.FusedPadConv2D'], lib="tf", suffix=0)
