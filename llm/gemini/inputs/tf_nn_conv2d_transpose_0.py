
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_tf_nn_conv2d_transpose_inputs():
    """
    Generates a list of valid inputs for the tf.nn.conv2d_transpose function.
    """
    list_of_inputs = []

    # Input 1: Basic NHWC, SAME padding, Stride 1
    input_dict_1 = {
        'input': np.ones((1, 4, 4, 2), dtype=np.float32),
        'filters': np.ones((3, 3, 3, 2), dtype=np.float32),
        'output_shape': np.array([1, 4, 4, 3], dtype=np.int32),
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'test_nhwc_same_s1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: NHWC, SAME padding, Stride 2
    input_dict_2 = {
        'input': np.ones((1, 4, 4, 2), dtype=np.float32),
        'filters': np.ones((3, 3, 3, 2), dtype=np.float32),
        'output_shape': np.array([1, 8, 8, 3], dtype=np.int32),
        'strides': [1, 2, 2, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'test_nhwc_same_s2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: NHWC, VALID padding, Stride 1
    input_dict_3 = {
        'input': np.zeros((1, 4, 4, 2), dtype=np.float32),
        'filters': np.zeros((3, 3, 3, 2), dtype=np.float32),
        'output_shape': np.array([1, 6, 6, 3], dtype=np.int32),
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'test_nhwc_valid_s1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: NHWC, VALID padding, Stride 2, larger batch
    input_dict_4 = {
        'input': np.ones((2, 5, 5, 1), dtype=np.float32),
        'filters': np.ones((2, 2, 4, 1), dtype=np.float32),
        'output_shape': np.array([2, 10, 10, 4], dtype=np.int32),
        'strides': [1, 2, 2, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'test_nhwc_valid_s2_batch2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Basic NCHW, SAME padding, Stride 1
    input_dict_5 = {
        'input': np.ones((1, 2, 4, 4), dtype=np.float32),
        'filters': np.ones((3, 3, 3, 2), dtype=np.float32),
        'output_shape': np.array([1, 3, 4, 4], dtype=np.int32),
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NCHW',
        'dilations': [1, 1, 1, 1],
        'name': 'test_nchw_same_s1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: NCHW, VALID padding, Stride 2
    input_dict_6 = {
        'input': np.ones((3, 1, 6, 6), dtype=np.float32),
        'filters': np.ones((3, 3, 5, 1), dtype=np.float32),
        'output_shape': np.array([3, 5, 13, 13], dtype=np.int32),
        'strides': [1, 1, 2, 2],
        'padding': 'VALID',
        'data_format': 'NCHW',
        'dilations': [1, 1, 1, 1],
        'name': 'test_nchw_valid_s2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: NHWC, SAME padding, Dilations > 1
    input_dict_7 = {
        'input': np.ones((1, 8, 8, 1), dtype=np.float32),
        'filters': np.ones((3, 3, 2, 1), dtype=np.float32),
        'output_shape': np.array([1, 8, 8, 2], dtype=np.int32),
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 2, 2, 1],
        'name': 'test_nhwc_same_dilation'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: NCHW, VALID padding, Dilations > 1
    input_dict_8 = {
        'input': np.ones((1, 3, 4, 4), dtype=np.float32),
        'filters': np.ones((3, 3, 2, 3), dtype=np.float32),
        'output_shape': np.array([1, 2, 11, 11], dtype=np.int32),
        'strides': [1, 1, 2, 2],
        'padding': 'VALID',
        'data_format': 'NCHW',
        'dilations': [1, 1, 2, 2],
        'name': 'test_nchw_valid_dilation'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Rectangular input and filter, NCHW, non-uniform strides
    input_dict_9 = {
        'input': np.ones((2, 4, 5, 10), dtype=np.float32),
        'filters': np.ones((3, 2, 6, 4), dtype=np.float32),
        'output_shape': np.array([2, 6, 10, 30], dtype=np.int32),
        'strides': [1, 1, 2, 3],
        'padding': 'SAME',
        'data_format': 'NCHW',
        'dilations': [1, 1, 1, 1],
        'name': 'test_nchw_rectangular'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Larger Batch Size, float64 type
    input_dict_10 = {
        'input': np.ones((16, 4, 4, 3), dtype=np.float64),
        'filters': np.ones((3, 3, 8, 3), dtype=np.float64),
        'output_shape': np.array([16, 8, 8, 8], dtype=np.int32),
        'strides': [1, 2, 2, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'test_large_batch_f64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Non-square filter
    input_dict_11 = {
        'input': np.ones((1, 8, 8, 2), dtype=np.float32),
        'filters': np.ones((1, 5, 4, 2), dtype=np.float32),
        'output_shape': np.array([1, 8, 12, 4], dtype=np.int32),
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'test_nonsquare_filter'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))
    
    return list_of_inputs

generated_inputs["tf.nn.conv2d_transpose"] = get_tf_nn_conv2d_transpose_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.conv2d_transpose' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.conv2d_transpose'.")

check_valid('tf.nn.conv2d_transpose', generated_inputs['tf.nn.conv2d_transpose'], lib="tf", suffix=0)
