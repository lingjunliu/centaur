
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_dilation2d_inputs():
    """
    This function generates a list of valid inputs for the tf.raw_ops.Dilation2D function.
    """
    list_of_inputs = []

    # Input 1: Basic float32, VALID padding
    input_dict_1 = {
        'input': np.random.rand(1, 5, 5, 1).astype(np.float32),
        'filter': np.random.rand(3, 3, 1).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'rates': [1, 1, 1, 1],
        'padding': 'VALID',
        'name': 'dilation_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Basic float32, SAME padding
    input_dict_2 = {
        'input': np.random.rand(1, 5, 5, 1).astype(np.float32),
        'filter': np.random.rand(3, 3, 1).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'rates': [1, 1, 1, 1],
        'padding': 'SAME',
        'name': 'dilation_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Strides > 1
    input_dict_3 = {
        'input': np.random.rand(1, 7, 7, 1).astype(np.float32),
        'filter': np.random.rand(3, 3, 1).astype(np.float32),
        'strides': [1, 2, 2, 1],
        'rates': [1, 1, 1, 1],
        'padding': 'VALID',
        'name': 'dilation_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Rates > 1 (atrous dilation)
    input_dict_4 = {
        'input': np.random.rand(1, 10, 10, 1).astype(np.float32),
        'filter': np.random.rand(3, 3, 1).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'rates': [1, 2, 2, 1],
        'padding': 'SAME',
        'name': 'dilation_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Multiple channels and batch size
    input_dict_5 = {
        'input': np.random.rand(4, 8, 8, 3).astype(np.float32),
        'filter': np.random.rand(3, 3, 3).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'rates': [1, 1, 1, 1],
        'padding': 'VALID',
        'name': 'dilation_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: int32 type with negative values
    input_dict_6 = {
        'input': np.random.randint(-50, 50, size=(1, 6, 6, 2)).astype(np.int32),
        'filter': np.random.randint(-10, 10, size=(2, 2, 2)).astype(np.int32),
        'strides': [1, 2, 2, 1],
        'rates': [1, 1, 1, 1],
        'padding': 'SAME',
        'name': 'dilation_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Max-pooling special case (zero filter) with uint8
    input_dict_7 = {
        'input': np.random.randint(0, 255, size=(1, 8, 8, 1), dtype=np.uint8),
        'filter': np.zeros((3, 3, 1), dtype=np.uint8),
        'strides': [1, 2, 2, 1],
        'rates': [1, 1, 1, 1],
        'padding': 'VALID',
        'name': 'dilation_7_maxpool'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Non-square filter and input
    input_dict_8 = {
        'input': np.random.rand(2, 7, 9, 1).astype(np.float64),
        'filter': np.random.rand(3, 2, 1).astype(np.float64),
        'strides': [1, 1, 1, 1],
        'rates': [1, 1, 1, 1],
        'padding': 'SAME',
        'name': 'dilation_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Different strides and rates
    input_dict_9 = {
        'input': np.arange(144).reshape(1, 12, 12, 1).astype(np.float32),
        'filter': np.zeros((3, 3, 1), dtype=np.float32),
        'strides': [1, 3, 2, 1],
        'rates': [1, 1, 2, 1],
        'padding': 'VALID',
        'name': 'dilation_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: int16 dtype
    input_dict_10 = {
        'input': np.random.randint(-1000, 1000, size=(1, 5, 5, 4)).astype(np.int16),
        'filter': np.random.randint(-100, 100, size=(3, 3, 4)).astype(np.int16),
        'strides': [1, 1, 1, 1],
        'rates': [1, 1, 1, 1],
        'padding': 'SAME',
        'name': 'dilation_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: 1x1 filter (identity for dilation with 0-filter)
    input_dict_11 = {
        'input': np.random.rand(1, 10, 10, 1).astype(np.float32),
        'filter': np.zeros((1, 1, 1), dtype=np.float32),
        'strides': [1, 1, 1, 1],
        'rates': [1, 1, 1, 1],
        'padding': 'VALID',
        'name': 'dilation_11_identity'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))
    
    # Input 12: Large filter
    input_dict_12 = {
        'input': np.random.rand(1, 10, 10, 2).astype(np.float32),
        'filter': np.random.rand(7, 7, 2).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'rates': [1, 1, 1, 1],
        'padding': 'VALID',
        'name': 'dilation_12_large_filter'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_12))

    return list_of_inputs

generated_inputs["tf.raw_ops.Dilation2D"] = tf_raw_ops_dilation2d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Dilation2D' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Dilation2D'.")

check_valid('tf.raw_ops.Dilation2D', generated_inputs['tf.raw_ops.Dilation2D'], lib="tf", suffix=0)
