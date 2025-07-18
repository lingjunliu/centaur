
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_nn_with_space_to_batch_inputs():
    """
    Generates a list of valid inputs for the tf.nn.with_space_to_batch function.
    """
    list_of_inputs = []

    def create_dummy_filter(rank):
        return np.ones([rank], dtype=np.int32)

    # Input 1: Basic 2D conv, NHWC, SAME padding, no dilation
    input_dict_1 = {
        'input': np.random.rand(1, 5, 5, 1).astype(np.float32),
        'dilation_rate': np.array([1, 1], dtype=np.int32),
        'padding': 'SAME',
        'op': ['tf.nn.conv2d'],
        'filter_shape': np.array([3, 3, 1, 1], dtype=np.int32),
        'spatial_dims': [1, 2],
        'data_format': 'NHWC'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Basic 2D conv, NHWC, VALID padding, no dilation
    input_dict_2 = {
        'input': np.random.rand(2, 6, 6, 3).astype(np.float32),
        'dilation_rate': np.array([1, 1], dtype=np.int32),
        'padding': 'VALID',
        'op': ['tf.nn.conv2d'],
        'filter_shape': create_dummy_filter(4),
        'spatial_dims': [1, 2],
        'data_format': 'NHWC'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Dilated 2D conv, NHWC, SAME padding
    input_dict_3 = {
        'input': np.random.rand(2, 10, 10, 3).astype(np.float32),
        'dilation_rate': np.array([2, 2], dtype=np.int32),
        'padding': 'SAME',
        'op': ['tf.nn.conv2d'],
        'filter_shape': np.array([3, 3, 3, 5], dtype=np.int32),
        'spatial_dims': [1, 2],
        'data_format': 'NHWC'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Dilated 2D conv, NCHW, SAME padding
    input_dict_4 = {
        'input': np.random.rand(2, 3, 10, 10).astype(np.float32),
        'dilation_rate': np.array([2, 2], dtype=np.int32),
        'padding': 'SAME',
        'op': ['tf.nn.conv2d'],
        'filter_shape': np.array([3, 3, 3, 5], dtype=np.int32),
        'spatial_dims': [2, 3],
        'data_format': 'NCHW'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Dilated 2D pooling, NHWC, VALID padding, non-uniform dilation
    input_dict_5 = {
        'input': np.random.rand(1, 20, 20, 1).astype(np.float32),
        'dilation_rate': np.array([3, 2], dtype=np.int32),
        'padding': 'VALID',
        'op': ['tf.nn.max_pool'],
        'filter_shape': create_dummy_filter(4),
        'spatial_dims': [1, 2],
        'data_format': 'NHWC'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: 1D conv, NWC, SAME padding
    input_dict_6 = {
        'input': np.random.rand(2, 50, 4).astype(np.float32),
        'dilation_rate': np.array([3], dtype=np.int32),
        'padding': 'SAME',
        'op': ['tf.nn.conv1d'],
        'filter_shape': np.array([5, 4, 8], dtype=np.int32),
        'spatial_dims': [1],
        'data_format': 'NWC'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: 1D conv, NCW, VALID padding
    input_dict_7 = {
        'input': np.random.rand(2, 4, 50).astype(np.float32),
        'dilation_rate': np.array([2], dtype=np.int32),
        'padding': 'VALID',
        'op': ['tf.nn.conv1d'],
        'filter_shape': create_dummy_filter(3),
        'spatial_dims': [2],
        'data_format': 'NCW'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: 3D conv, NDHWC, SAME padding
    input_dict_8 = {
        'input': np.random.rand(1, 8, 8, 8, 1).astype(np.float32),
        'dilation_rate': np.array([2, 2, 2], dtype=np.int32),
        'padding': 'SAME',
        'op': ['tf.nn.conv3d'],
        'filter_shape': np.array([3, 3, 3, 1, 4], dtype=np.int32),
        'spatial_dims': [1, 2, 3],
        'data_format': 'NDHWC'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: 3D conv, NCDHW, VALID padding, non-uniform dilation
    input_dict_9 = {
        'input': np.random.rand(1, 2, 8, 8, 8).astype(np.float32),
        'dilation_rate': np.array([1, 2, 3], dtype=np.int32),
        'padding': 'VALID',
        'op': ['tf.nn.conv3d'],
        'filter_shape': create_dummy_filter(5),
        'spatial_dims': [2, 3, 4],
        'data_format': 'NCDHW'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: 2D depthwise conv, NHWC, SAME padding, non-uniform dilation
    input_dict_10 = {
        'input': np.random.rand(4, 7, 9, 2).astype(np.float32),
        'dilation_rate': np.array([2, 1], dtype=np.int32),
        'padding': 'SAME',
        'op': ['tf.nn.depthwise_conv2d'],
        'filter_shape': np.array([3, 3, 2, 1], dtype=np.int32),
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
