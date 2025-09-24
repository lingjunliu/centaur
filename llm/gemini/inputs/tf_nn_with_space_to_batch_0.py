
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_nn_with_space_to_batch_inputs():
    """
    Generates a list of valid inputs for tf.nn.with_space_to_batch.
    """
    list_of_inputs = []

    # A simple op function that satisfies the required signature.
    # To satisfy the testing harness which expects an iterable, it is wrapped in a list.
    op_func = lambda x, num_spatial_dims, padding: x

    # Case 1: Basic 2D Convolution scenario (NHWC, SAME padding)
    input_dict_1 = {
        'input': np.random.rand(1, 5, 5, 3).astype(np.float32),
        'dilation_rate': np.array([2, 2], dtype=np.int32),
        'padding': 'SAME',
        'op': [op_func],
        'filter_shape': np.array([3, 3], dtype=np.int32),
        'spatial_dims': [1, 2],
        'data_format': 'NHWC'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Case 2: Basic 2D with VALID padding
    input_dict_2 = {
        'input': np.random.rand(2, 8, 8, 1).astype(np.float32),
        'dilation_rate': np.array([3, 3], dtype=np.int32),
        'padding': 'VALID',
        'op': [op_func],
        'filter_shape': np.array([1, 1], dtype=np.int32),
        'spatial_dims': [1, 2],
        'data_format': 'NHWC'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Case 3: 1D Spatial dimension (e.g., Conv1D on NWC data)
    input_dict_3 = {
        'input': np.arange(4 * 16 * 2, dtype=np.float32).reshape(4, 16, 2),
        'dilation_rate': np.array([3], dtype=np.int32),
        'padding': 'SAME',
        'op': [op_func],
        'filter_shape': np.array([3], dtype=np.int32),
        'spatial_dims': [1],
        'data_format': 'NWC'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Case 4: 3D Spatial dimensions (NDHWC, e.g., Conv3D)
    input_dict_4 = {
        'input': np.ones((1, 4, 4, 4, 2), dtype=np.float32),
        'dilation_rate': np.array([2, 1, 2], dtype=np.int32),
        'padding': 'SAME',
        'op': [op_func],
        'filter_shape': np.array([3, 3, 3], dtype=np.int32),
        'spatial_dims': [1, 2, 3],
        'data_format': 'NDHWC'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Case 5: NCHW data format
    input_dict_5 = {
        'input': np.random.rand(2, 3, 7, 7).astype(np.float32),
        'dilation_rate': np.array([2, 3], dtype=np.int32),
        'padding': 'SAME',
        'op': [op_func],
        'filter_shape': np.array([3, 3], dtype=np.int32),
        'spatial_dims': [2, 3],
        'data_format': 'NCHW'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Case 6: NCDHW data format with VALID padding
    input_dict_6 = {
        'input': np.zeros((1, 2, 5, 5, 5), dtype=np.float32),
        'dilation_rate': np.array([1, 2, 3], dtype=np.int32),
        'padding': 'VALID',
        'op': [op_func],
        'filter_shape': np.array([1, 1, 1], dtype=np.int32),
        'spatial_dims': [2, 3, 4],
        'data_format': 'NCDHW'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Case 7: Special case with uniform dilation_rate of 1
    input_dict_7 = {
        'input': np.random.rand(4, 10, 10, 1).astype(np.float32),
        'dilation_rate': np.array([1, 1], dtype=np.int32),
        'padding': 'VALID',
        'op': [op_func],
        'filter_shape': np.array([2, 2], dtype=np.int32),
        'spatial_dims': [1, 2],
        'data_format': 'NHWC'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Case 8: Non-square filter and non-uniform dilation
    input_dict_8 = {
        'input': np.random.rand(1, 10, 12, 3).astype(np.float32),
        'dilation_rate': np.array([2, 3], dtype=np.int32),
        'padding': 'SAME',
        'op': [op_func],
        'filter_shape': np.array([3, 5], dtype=np.int32),
        'spatial_dims': [1, 2],
        'data_format': 'NHWC'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Case 9: NCW data format
    input_dict_9 = {
        'input': np.arange(4*2*16).reshape(4, 2, 16).astype(np.float32),
        'dilation_rate': np.array([2], dtype=np.int32),
        'padding': 'SAME',
        'op': [op_func],
        'filter_shape': np.array([3], dtype=np.int32),
        'spatial_dims': [2],
        'data_format': 'NCW'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Case 10: High dilation rate
    input_dict_10 = {
        'input': np.random.rand(1, 16, 16, 3).astype(np.float32),
        'dilation_rate': np.array([4, 4], dtype=np.int32),
        'padding': 'SAME',
        'op': [op_func],
        'filter_shape': np.array([3, 3], dtype=np.int32),
        'spatial_dims': [1, 2],
        'data_format': 'NHWC'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.nn.with_space_to_batch"] = tf_nn_with_space_to_batch_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.with_space_to_batch' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.with_space_to_batch'.")

check_valid('tf.nn.with_space_to_batch', generated_inputs['tf.nn.with_space_to_batch'], lib="tf", suffix=0)
