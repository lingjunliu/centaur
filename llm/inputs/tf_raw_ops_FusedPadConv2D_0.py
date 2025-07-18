
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_fusedpadconv2d_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.FusedPadConv2D function.
    """
    list_of_inputs = []

    # Case 1: Basic VALID padding, float32, REFLECT mode
    input_dict_1 = {
        'input': np.random.rand(1, 5, 5, 1).astype(np.float32),
        'paddings': np.array([[0, 0], [1, 1], [1, 1], [0, 0]], dtype=np.int32),
        'filter': np.random.rand(3, 3, 1, 2).astype(np.float32),
        'mode': 'REFLECT',
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'name': 'case1_valid_reflect'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Case 2: Basic SAME padding, float32, SYMMETRIC mode
    input_dict_2 = {
        'input': np.random.rand(2, 6, 6, 3).astype(np.float32),
        'paddings': np.array([[0, 0], [2, 2], [2, 2], [0, 0]], dtype=np.int32),
        'filter': np.random.rand(3, 3, 3, 4).astype(np.float32),
        'mode': 'SYMMETRIC',
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'name': 'case2_same_symmetric'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Case 3: Strides > 1, float64
    input_dict_3 = {
        'input': np.random.rand(1, 8, 8, 2).astype(np.float64),
        'paddings': np.array([[0, 0], [1, 1], [1, 1], [0, 0]], dtype=np.int32),
        'filter': np.random.rand(2, 2, 2, 5).astype(np.float64),
        'mode': 'REFLECT',
        'strides': [1, 2, 2, 1],
        'padding': 'VALID',
        'name': 'case3_strided_valid'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Case 4: float16 (half) type
    input_dict_4 = {
        'input': np.random.rand(1, 7, 7, 1).astype(np.float16),
        'paddings': np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.int32),
        'filter': np.random.rand(3, 3, 1, 2).astype(np.float16),
        'mode': 'SYMMETRIC',
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'name': 'case4_float16_valid'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Case 5: Larger batch size
    input_dict_5 = {
        'input': np.random.rand(4, 5, 5, 1).astype(np.float32),
        'paddings': np.array([[0, 0], [1, 1], [1, 1], [0, 0]], dtype=np.int32),
        'filter': np.random.rand(3, 3, 1, 2).astype(np.float32),
        'mode': 'SYMMETRIC',
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'name': 'case6_large_batch'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Case 6: Non-square input/filter
    input_dict_6 = {
        'input': np.random.rand(1, 5, 7, 2).astype(np.float32),
        'paddings': np.array([[0, 0], [1, 1], [2, 2], [0, 0]], dtype=np.int32),
        'filter': np.random.rand(2, 4, 2, 3).astype(np.float32),
        'mode': 'REFLECT',
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'name': 'case7_nonsquare'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Case 7: Different strides in H and W
    input_dict_7 = {
        'input': np.random.rand(1, 9, 9, 1).astype(np.float64),
        'paddings': np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.int32),
        'filter': np.random.rand(3, 3, 1, 1).astype(np.float64),
        'mode': 'SYMMETRIC',
        'strides': [1, 1, 2, 1],
        'padding': 'VALID',
        'name': 'case8_nonsquare_stride'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Case 8: No padding provided (paddings are all zeros)
    input_dict_8 = {
        'input': np.random.rand(1, 4, 4, 3).astype(np.float32),
        'paddings': np.zeros((4, 2), dtype=np.int32),
        'filter': np.random.rand(2, 2, 3, 5).astype(np.float32),
        'mode': 'REFLECT',
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'name': 'case9_no_padding'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Case 9: Large paddings
    input_dict_9 = {
        'input': np.random.rand(1, 3, 3, 1).astype(np.float32),
        'paddings': np.array([[0, 0], [3, 3], [3, 3], [0, 0]], dtype=np.int32),
        'filter': np.random.rand(3, 3, 1, 1).astype(np.float32),
        'mode': 'SYMMETRIC',
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'name': 'case10_large_padding'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Case 10: 1x1 filter (pointwise convolution)
    input_dict_10 = {
        'input': np.random.rand(2, 8, 8, 16).astype(np.float32),
        'paddings': np.array([[0, 0], [1, 1], [1, 1], [0, 0]], dtype=np.int32),
        'filter': np.random.rand(1, 1, 16, 32).astype(np.float32),
        'mode': 'REFLECT',
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'name': 'case11_pointwise'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.FusedPadConv2D"] = get_fusedpadconv2d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.FusedPadConv2D' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FusedPadConv2D'.")

check_valid('tf.raw_ops.FusedPadConv2D', generated_inputs['tf.raw_ops.FusedPadConv2D'], lib="tf", suffix=0)
