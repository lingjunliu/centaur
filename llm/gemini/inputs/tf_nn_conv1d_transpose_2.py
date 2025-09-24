
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_conv1d_transpose_inputs():
    """
    Generates a list of valid inputs for the tf.nn.conv1d_transpose function.
    """
    list_of_inputs = []

    # Case 1: Basic NWC, SAME padding, stride 1
    input_1 = np.random.randn(1, 5, 3).astype(np.float32)
    filters_1 = np.random.randn(3, 8, 3).astype(np.float32)
    # For SAME padding, output_width = input_width * stride = 5 * 1 = 5
    output_shape_1 = np.array([1, 5, 8], dtype=np.int32)
    input_dict_1 = {
        'input': input_1,
        'filters': filters_1,
        'output_shape': output_shape_1,
        'strides': [1],
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': [1],
        'name': 'case_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Case 2: Basic NWC, VALID padding, stride 1
    input_2 = np.random.randn(1, 5, 3).astype(np.float32)
    filters_2 = np.random.randn(3, 8, 3).astype(np.float32)
    # For VALID padding, output_width = (input_width - 1) * stride + filter_width = (5 - 1) * 1 + 3 = 7
    output_shape_2 = np.array([1, 7, 8], dtype=np.int32)
    input_dict_2 = {
        'input': input_2,
        'filters': filters_2,
        'output_shape': output_shape_2,
        'strides': [1],
        'padding': 'VALID',
        'data_format': 'NWC',
        'dilations': [1],
        'name': 'case_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Case 3: NWC, SAME padding, stride > 1
    input_3 = np.random.randn(2, 4, 2).astype(np.float32)
    filters_3 = np.random.randn(2, 4, 2).astype(np.float32)
    # For SAME padding, output_width = input_width * stride = 4 * 2 = 8
    output_shape_3 = np.array([2, 8, 4], dtype=np.int32)
    input_dict_3 = {
        'input': input_3,
        'filters': filters_3,
        'output_shape': output_shape_3,
        'strides': [2],
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': [1],
        'name': 'case_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Case 4: NWC, VALID padding, stride > 1
    input_4 = np.random.randn(2, 4, 2).astype(np.float32)
    filters_4 = np.random.randn(2, 4, 2).astype(np.float32)
    # For VALID padding, output_width = (input_width - 1) * stride + filter_width = (4 - 1) * 2 + 2 = 8
    output_shape_4 = np.array([2, 8, 4], dtype=np.int32)
    input_dict_4 = {
        'input': input_4,
        'filters': filters_4,
        'output_shape': output_shape_4,
        'strides': [2],
        'padding': 'VALID',
        'data_format': 'NWC',
        'dilations': [1],
        'name': 'case_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Case 5: NCW format, SAME padding, stride 1
    input_5 = np.random.randn(1, 3, 5).astype(np.float32)
    filters_5 = np.random.randn(3, 8, 3).astype(np.float32)
    # output_shape for NCW: [batch, output_channels, output_width]
    # For SAME padding, output_width = input_width * stride = 5 * 1 = 5
    output_shape_5 = np.array([1, 8, 5], dtype=np.int32)
    input_dict_5 = {
        'input': input_5,
        'filters': filters_5,
        'output_shape': output_shape_5,
        'strides': [1],
        'padding': 'SAME',
        'data_format': 'NCW',
        'dilations': [1],
        'name': 'case_5_ncw'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Case 6: NCW format, VALID padding, stride > 1
    input_6 = np.random.randn(2, 2, 4).astype(np.float32)
    filters_6 = np.random.randn(2, 4, 2).astype(np.float32)
    # output_shape for NCW: [batch, output_channels, output_width]
    # For VALID padding, output_width = (input_width - 1) * stride + filter_width = (4 - 1) * 2 + 2 = 8
    output_shape_6 = np.array([2, 4, 8], dtype=np.int32)
    input_dict_6 = {
        'input': input_6,
        'filters': filters_6,
        'output_shape': output_shape_6,
        'strides': [2],
        'padding': 'VALID',
        'data_format': 'NCW',
        'dilations': [1],
        'name': 'case_6_ncw'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Case 7: NWC, with dilations > 1
    input_7 = np.random.randn(1, 5, 3).astype(np.float32)
    filters_7 = np.random.randn(3, 8, 3).astype(np.float32)
    # For SAME padding with stride=2, output_width = input_width * stride = 5 * 2 = 10
    output_shape_7 = np.array([1, 10, 8], dtype=np.int32)
    input_dict_7 = {
        'input': input_7,
        'filters': filters_7,
        'output_shape': output_shape_7,
        'strides': [2],
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': [2],
        'name': 'case_7_dilations'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Case 8: Full 3-element strides and dilations, NWC
    input_8 = np.random.randn(1, 6, 4).astype(np.float32)
    filters_8 = np.random.randn(4, 2, 4).astype(np.float32)
    # Spatial stride is 2. output_width = input_width * stride = 6 * 2 = 12
    output_shape_8 = np.array([1, 12, 2], dtype=np.int32)
    input_dict_8 = {
        'input': input_8,
        'filters': filters_8,
        'output_shape': output_shape_8,
        'strides': [1, 2, 1],
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': [1, 2, 1],
        'name': 'case_8_full_strides_dilations'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Case 9: Larger batch and channels, NWC
    input_9 = np.random.randn(4, 10, 16).astype(np.float32)
    filters_9 = np.random.randn(5, 32, 16).astype(np.float32)
    # For VALID padding, output_width = (10 - 1) * 3 + 5 = 32
    output_shape_9 = np.array([4, 32, 32], dtype=np.int32)
    input_dict_9 = {
        'input': input_9,
        'filters': filters_9,
        'output_shape': output_shape_9,
        'strides': [3],
        'padding': 'VALID',
        'data_format': 'NWC',
        'dilations': [1],
        'name': 'case_9_large'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Case 10: Minimalist case, 1x1 input, 1x1 filter
    input_10 = np.ones((1, 1, 1), dtype=np.float32)
    filters_10 = np.ones((1, 1, 1), dtype=np.float32)
    output_shape_10 = np.array([1, 1, 1], dtype=np.int32)
    input_dict_10 = {
        'input': input_10,
        'filters': filters_10,
        'output_shape': output_shape_10,
        'strides': [1],
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': [1],
        'name': 'case_10_minimal'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Case 11: Mismatched output_shape (but valid) for 'SAME' padding
    input_11 = np.random.randn(1, 5, 3).astype(np.float32)
    filters_11 = np.random.randn(3, 8, 3).astype(np.float32)
    # Standard calculation: output_width = 5 * 2 = 10.
    # A slightly smaller output_shape is also valid.
    output_shape_11 = np.array([1, 9, 8], dtype=np.int32)
    input_dict_11 = {
        'input': input_11,
        'filters': filters_11,
        'output_shape': output_shape_11,
        'strides': [2],
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': [1],
        'name': 'case_11_mismatched_shape'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    return list_of_inputs

generated_inputs["tf.nn.conv1d_transpose_2"] = get_conv1d_transpose_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.conv1d_transpose_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.conv1d_transpose_2'.")

check_valid('tf.nn.conv1d_transpose', generated_inputs['tf.nn.conv1d_transpose_2'], lib="tf", suffix=2)
