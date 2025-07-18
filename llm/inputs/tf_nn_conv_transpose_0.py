
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_nn_conv_transpose_inputs():
    """
    This function generates a list of valid inputs for the tf.nn.conv_transpose API.
    The filter shape is [filter_spatial_dims, out_channels, in_channels].
    """
    list_of_inputs = []

    # Input 1: Basic 2D, NHWC, SAME padding, Stride 1
    list_of_inputs.append(copy.deepcopy({
        'input': np.random.rand(1, 8, 8, 3).astype(np.float32),
        'filters': np.random.rand(3, 3, 5, 3).astype(np.float32),
        'output_shape': np.array([1, 8, 8, 5], dtype=np.int32),
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'basic_2d_same'
    }))

    # Input 2: Basic 2D, NHWC, VALID padding, Stride 1
    list_of_inputs.append(copy.deepcopy({
        'input': np.random.rand(1, 8, 8, 3).astype(np.float32),
        'filters': np.random.rand(3, 3, 5, 3).astype(np.float32),
        'output_shape': np.array([1, 10, 10, 5], dtype=np.int32),
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'basic_2d_valid'
    }))

    # Input 3: Strided 2D (Upsampling), NHWC, SAME padding
    list_of_inputs.append(copy.deepcopy({
        'input': np.random.rand(1, 8, 8, 3).astype(np.float32),
        'filters': np.random.rand(3, 3, 5, 3).astype(np.float32),
        'output_shape': np.array([1, 16, 16, 5], dtype=np.int32),
        'strides': [1, 2, 2, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'strided_2d_same'
    }))

    # Input 4: Strided 2D (Upsampling), NHWC, VALID padding
    list_of_inputs.append(copy.deepcopy({
        'input': np.random.rand(1, 8, 8, 3).astype(np.float32),
        'filters': np.random.rand(3, 3, 5, 3).astype(np.float32),
        'output_shape': np.array([1, 17, 17, 5], dtype=np.int32),
        'strides': [1, 2, 2, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'strided_2d_valid'
    }))

    # Input 5: Dilated 2D, NHWC, SAME padding
    list_of_inputs.append(copy.deepcopy({
        'input': np.random.rand(1, 8, 8, 3).astype(np.float32),
        'filters': np.random.rand(3, 3, 5, 3).astype(np.float32),
        'output_shape': np.array([1, 8, 8, 5], dtype=np.int32),
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 2, 2, 1],
        'name': 'dilated_2d_same'
    }))

    # Input 6: Basic 2D, NCHW, SAME padding
    list_of_inputs.append(copy.deepcopy({
        'input': np.random.rand(1, 3, 8, 8).astype(np.float32),
        'filters': np.random.rand(3, 3, 5, 3).astype(np.float32),
        'output_shape': np.array([1, 5, 8, 8], dtype=np.int32),
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NCHW',
        'dilations': [1, 1, 1, 1],
        'name': 'basic_2d_nchw'
    }))

    # Input 7: Strided 2D, NCHW, SAME padding
    list_of_inputs.append(copy.deepcopy({
        'input': np.random.rand(1, 3, 8, 8).astype(np.float32),
        'filters': np.random.rand(3, 3, 5, 3).astype(np.float32),
        'output_shape': np.array([1, 5, 16, 16], dtype=np.int32),
        'strides': [1, 1, 2, 2],
        'padding': 'SAME',
        'data_format': 'NCHW',
        'dilations': [1, 1, 1, 1],
        'name': 'strided_2d_nchw'
    }))

    # Input 8: Basic 1D, NWC, SAME padding, Stride 2
    list_of_inputs.append(copy.deepcopy({
        'input': np.random.rand(2, 16, 4).astype(np.float32),
        'filters': np.random.rand(3, 8, 4).astype(np.float32),
        'output_shape': np.array([2, 32, 8], dtype=np.int32),
        'strides': [1, 2, 1],
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': [1, 1, 1],
        'name': 'basic_1d_nwc'
    }))

    # Input 9: Basic 3D, NDHWC, SAME padding, Stride 2
    list_of_inputs.append(copy.deepcopy({
        'input': np.random.rand(1, 4, 4, 4, 2).astype(np.float32),
        'filters': np.random.rand(3, 3, 3, 3, 2).astype(np.float32),
        'output_shape': np.array([1, 8, 8, 8, 3], dtype=np.int32),
        'strides': [1, 2, 2, 2, 1],
        'padding': 'SAME',
        'data_format': 'NDHWC',
        'dilations': [1, 1, 1, 1, 1],
        'name': 'basic_3d_ndhwc'
    }))

    # Input 10: float16 dtype, 2D, NHWC, SAME padding, Stride 2
    list_of_inputs.append(copy.deepcopy({
        'input': np.random.rand(1, 8, 8, 3).astype(np.float16),
        'filters': np.random.rand(3, 3, 5, 3).astype(np.float16),
        'output_shape': np.array([1, 16, 16, 5], dtype=np.int32),
        'strides': [1, 2, 2, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'float16_2d_same'
    }))

    return list_of_inputs

generated_inputs["tf.nn.conv_transpose"] = tf_nn_conv_transpose_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.conv_transpose' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.conv_transpose'.")

check_valid('tf.nn.conv_transpose', generated_inputs['tf.nn.conv_transpose'], lib="tf", suffix=0)
