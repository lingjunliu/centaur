
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf

# Define callable list classes to satisfy both pre-processing (expects a list)
# and runtime (expects a callable). These classes inherit from `list` so they
# are considered lists by type checkers and `isinstance`, but they also
# implement `__call__` to be used as functions.

class OpConv1d(list):
    """A callable list that mimics a 1D convolution operation."""
    def __init__(self):
        super().__init__([0]) # Content for pre-processing script
    def __call__(self, input_tensor, _, padding):
        # The op is called on the transformed tensor, which has channels last.
        filter_shape = [3, input_tensor.shape[-1], 1]
        filt = tf.zeros(filter_shape, dtype=input_tensor.dtype)
        return tf.nn.conv1d(input_tensor, filt, stride=1, padding=padding.upper())

class OpConv2d(list):
    """A callable list that mimics a 2D convolution operation."""
    def __init__(self):
        super().__init__([0]) # Content for pre-processing script
    def __call__(self, input_tensor, _, padding):
        # The op is called on the transformed tensor, which has channels last.
        filter_shape = [3, 3, input_tensor.shape[-1], 1]
        filt = tf.zeros(filter_shape, dtype=input_tensor.dtype)
        return tf.nn.conv2d(input_tensor, filt, strides=[1, 1, 1, 1], padding=padding.upper())

class OpConv3d(list):
    """A callable list that mimics a 3D convolution operation."""
    def __init__(self):
        super().__init__([0]) # Content for pre-processing script
    def __call__(self, input_tensor, _, padding):
        # The op is called on the transformed tensor, which has channels last.
        filter_shape = [3, 3, 3, input_tensor.shape[-1], 1]
        filt = tf.zeros(filter_shape, dtype=input_tensor.dtype)
        return tf.nn.conv3d(input_tensor, filt, strides=[1, 1, 1, 1, 1], padding=padding.upper())

class OpMaxPool2d(list):
    """A callable list that mimics a 2D max pooling operation."""
    def __init__(self):
        super().__init__([1]) # Content for pre-processing script
    def __call__(self, input_tensor, _, padding):
        # The op is called on the transformed tensor, which has channels last.
        ksize = [1, 2, 2, 1]
        strides = [1, 1, 1, 1]
        return tf.nn.max_pool(input_tensor, ksize, strides, padding=padding.upper())


def tf_nn_with_space_to_batch_inputs():
    """
    Generates a list of valid inputs for tf.nn.with_space_to_batch.
    """
    list_of_inputs = []

    # Input 1: Basic 2D 'SAME' padding, NHWC format with conv2d-like op
    input_dict_1 = {
        'input': np.random.rand(2, 8, 8, 3).astype(np.float32),
        'dilation_rate': np.array([2, 2], dtype=np.int32),
        'padding': "SAME",
        'op': OpConv2d(),
        'filter_shape': np.array([3, 3], dtype=np.int32),
        'spatial_dims': [1, 2],
        'data_format': "NHWC"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D 'VALID' padding with max_pool-like op
    input_dict_2 = {
        'input': np.random.rand(1, 10, 10, 1).astype(np.float32),
        'dilation_rate': np.array([3, 3], dtype=np.int32),
        'padding': "VALID",
        'op': OpMaxPool2d(),
        'filter_shape': np.array([2, 2], dtype=np.int32),
        'spatial_dims': [1, 2],
        'data_format': "NHWC"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Special case - dilation rate is uniformly 1
    input_dict_3 = {
        'input': np.random.rand(4, 12, 12, 2).astype(np.float32),
        'dilation_rate': np.array([1, 1], dtype=np.int32),
        'padding': "SAME",
        'op': OpConv2d(),
        'filter_shape': np.array([3, 3], dtype=np.int32),
        'spatial_dims': [1, 2],
        'data_format': "NHWC"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: NCHW data format
    input_dict_4 = {
        'input': np.random.rand(2, 3, 8, 8).astype(np.float32),
        'dilation_rate': np.array([2, 2], dtype=np.int32),
        'padding': "SAME",
        'op': OpConv2d(),
        'filter_shape': np.array([3, 3], dtype=np.int32),
        'spatial_dims': [2, 3],
        'data_format': "NCHW"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: 1D case (e.g., for conv1d)
    input_dict_5 = {
        'input': np.random.rand(4, 20, 2).astype(np.float32),
        'dilation_rate': np.array([4], dtype=np.int32),
        'padding': "SAME",
        'op': OpConv1d(),
        'filter_shape': np.array([3], dtype=np.int32),
        'spatial_dims': [1],
        'data_format': "NWC"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: 3D case (e.g., for conv3d) NDHWC format
    input_dict_6 = {
        'input': np.random.rand(2, 6, 6, 6, 3).astype(np.float32),
        'dilation_rate': np.array([2, 2, 2], dtype=np.int32),
        'padding': "SAME",
        'op': OpConv3d(),
        'filter_shape': np.array([3, 3, 3], dtype=np.int32),
        'spatial_dims': [1, 2, 3],
        'data_format': "NDHWC"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: non-uniform dilation rate
    input_dict_7 = {
        'input': np.random.rand(2, 9, 9, 3).astype(np.float32),
        'dilation_rate': np.array([2, 3], dtype=np.int32),
        'padding': "SAME",
        'op': OpConv2d(),
        'filter_shape': np.array([3, 3], dtype=np.int32),
        'spatial_dims': [1, 2],
        'data_format': "NHWC"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: 1D case with NCW format
    input_dict_8 = {
        'input': np.random.rand(4, 2, 20).astype(np.float32),
        'dilation_rate': np.array([3], dtype=np.int32),
        'padding': "SAME",
        'op': OpConv1d(),
        'filter_shape': np.array([3], dtype=np.int32),
        'spatial_dims': [2],
        'data_format': "NCW"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: 3D case with NCDHW format
    input_dict_9 = {
        'input': np.random.rand(1, 2, 8, 8, 8).astype(np.float32),
        'dilation_rate': np.array([2, 2, 2], dtype=np.int32),
        'padding': "SAME",
        'op': OpConv3d(),
        'filter_shape': np.array([3, 3, 3], dtype=np.int32),
        'spatial_dims': [2, 3, 4],
        'data_format': "NCDHW"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: VALID padding with non-uniform dilation
    input_dict_10 = {
        'input': np.random.rand(1, 15, 12, 1).astype(np.float32),
        'dilation_rate': np.array([4, 2], dtype=np.int32),
        'padding': "VALID",
        'op': OpConv2d(),
        'filter_shape': np.array([], dtype=np.int32), # ignored for VALID
        'spatial_dims': [1, 2],
        'data_format': "NHWC"
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
