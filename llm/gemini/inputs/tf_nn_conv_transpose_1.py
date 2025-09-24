
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_tf_nn_conv_transpose_inputs():
    """
    Generates a list of valid inputs for the tf.nn.conv_transpose function.
    """
    list_of_inputs = []

    # Input 1: Basic 2D transpose convolution, NHWC format, 'SAME' padding
    input_1 = np.random.rand(1, 4, 4, 1).astype(np.float32)
    filters_1 = np.random.rand(3, 3, 2, 1).astype(np.float32)
    output_shape_1 = np.array([1, 8, 8, 2], dtype=np.int32)
    input_dict_1 = {
        'input': input_1,
        'filters': filters_1,
        'output_shape': output_shape_1,
        'strides': 2,
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': 1,
        'name': 'conv_transpose_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D transpose convolution, NHWC format, 'VALID' padding
    input_2 = np.random.rand(1, 4, 4, 1).astype(np.float32)
    filters_2 = np.random.rand(3, 3, 2, 1).astype(np.float32)
    output_shape_2 = np.array([1, 9, 9, 2], dtype=np.int32)
    input_dict_2 = {
        'input': input_2,
        'filters': filters_2,
        'output_shape': output_shape_2,
        'strides': 2,
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': 1,
        'name': 'conv_transpose_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 2D transpose convolution, NCHW format, 'SAME' padding
    input_3 = np.random.rand(1, 3, 10, 12).astype(np.float32)
    filters_3 = np.random.rand(5, 5, 6, 3).astype(np.float32)
    output_shape_3 = np.array([1, 6, 10, 12], dtype=np.int32)
    input_dict_3 = {
        'input': input_3,
        'filters': filters_3,
        'output_shape': output_shape_3,
        'strides': 1,
        'padding': 'SAME',
        'data_format': 'NCHW',
        'dilations': 1,
        'name': 'conv_transpose_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 2D with non-unit stride and dilation
    input_4 = np.random.rand(2, 5, 5, 2).astype(np.float32)
    filters_4 = np.random.rand(3, 3, 4, 2).astype(np.float32)
    output_shape_4 = np.array([2, 15, 15, 4], dtype=np.int32)
    input_dict_4 = {
        'input': input_4,
        'filters': filters_4,
        'output_shape': output_shape_4,
        'strides': 3,
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': 2,
        'name': 'conv_transpose_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: 1D transpose convolution, NWC format
    input_5 = np.random.rand(1, 10, 3).astype(np.float32)
    filters_5 = np.random.rand(5, 4, 3).astype(np.float32)
    output_shape_5 = np.array([1, 20, 4], dtype=np.int32)
    input_dict_5 = {
        'input': input_5,
        'filters': filters_5,
        'output_shape': output_shape_5,
        'strides': 2,
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': 1,
        'name': 'conv1d_transpose_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: 1D transpose convolution, NCW format, 'VALID' padding
    input_6 = np.random.rand(2, 5, 16).astype(np.float32)
    filters_6 = np.random.rand(3, 8, 5).astype(np.float32)
    output_shape_6 = np.array([2, 8, 18], dtype=np.int32)
    input_dict_6 = {
        'input': input_6,
        'filters': filters_6,
        'output_shape': output_shape_6,
        'strides': 1,
        'padding': 'VALID',
        'data_format': 'NCW',
        'dilations': 1,
        'name': 'conv1d_transpose_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: 3D transpose convolution, NDHWC format, 'SAME' padding
    input_7 = np.random.rand(1, 4, 4, 4, 2).astype(np.float32)
    filters_7 = np.random.rand(3, 3, 3, 3, 2).astype(np.float32)
    output_shape_7 = np.array([1, 8, 8, 8, 3], dtype=np.int32)
    input_dict_7 = {
        'input': input_7,
        'filters': filters_7,
        'output_shape': output_shape_7,
        'strides': 2,
        'padding': 'SAME',
        'data_format': 'NDHWC',
        'dilations': 1,
        'name': 'conv3d_transpose_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: 3D transpose convolution, NCDHW format, 'VALID' padding
    input_8 = np.random.rand(1, 2, 5, 6, 7).astype(np.float32)
    filters_8 = np.random.rand(3, 3, 3, 4, 2).astype(np.float32)
    output_shape_8 = np.array([1, 4, 7, 8, 9], dtype=np.int32)
    input_dict_8 = {
        'input': input_8,
        'filters': filters_8,
        'output_shape': output_shape_8,
        'strides': 1,
        'padding': 'VALID',
        'data_format': 'NCDHW',
        'dilations': 1,
        'name': 'conv3d_transpose_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: float64 data type
    input_9 = np.random.rand(1, 4, 4, 1).astype(np.float64)
    filters_9 = np.random.rand(3, 3, 2, 1).astype(np.float64)
    output_shape_9 = np.array([1, 8, 8, 2], dtype=np.int32)
    input_dict_9 = {
        'input': input_9,
        'filters': filters_9,
        'output_shape': output_shape_9,
        'strides': 2,
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': 1,
        'name': 'conv_transpose_float64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Another 2D case with different dimensions and 'VALID' padding
    input_10 = np.random.rand(1, 7, 6, 3).astype(np.float32)
    filters_10 = np.random.rand(4, 4, 5, 3).astype(np.float32)
    output_shape_10 = np.array([1, 16, 14, 5], dtype=np.int32)
    input_dict_10 = {
        'input': input_10,
        'filters': filters_10,
        'output_shape': output_shape_10,
        'strides': 2,
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': 1,
        'name': 'conv_transpose_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.nn.conv_transpose_1"] = get_tf_nn_conv_transpose_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.conv_transpose_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.conv_transpose_1'.")

check_valid('tf.nn.conv_transpose', generated_inputs['tf.nn.conv_transpose_1'], lib="tf", suffix=1)
