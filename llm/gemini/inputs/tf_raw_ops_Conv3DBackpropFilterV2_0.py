
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def _get_out_size_conv3d(in_size, filter_size, stride, padding, dilation=1):
    """Helper function to calculate the output dimension of a 3D convolution."""
    if padding.upper() == 'SAME':
        return (in_size + stride - 1) // stride
    elif padding.upper() == 'VALID':
        effective_filter_size = (filter_size - 1) * dilation + 1
        return (in_size - effective_filter_size) // stride + 1
    return 0

def tf_raw_ops_conv3dbackpropfilterv2_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.Conv3DBackpropFilterV2.
    """
    list_of_inputs = []

    # Input 1: Basic case, NDHWC, VALID padding, float32
    input_shape_1 = (2, 5, 6, 7, 3)
    filter_sizes_1 = np.array([3, 3, 3, 3, 4], dtype=np.int32)
    strides_1 = [1, 1, 1, 1, 1]
    padding_1 = "VALID"
    dilations_1 = [1, 1, 1, 1, 1]
    out_d_1 = _get_out_size_conv3d(input_shape_1[1], filter_sizes_1[0], strides_1[1], padding_1, dilations_1[1])
    out_h_1 = _get_out_size_conv3d(input_shape_1[2], filter_sizes_1[1], strides_1[2], padding_1, dilations_1[2])
    out_w_1 = _get_out_size_conv3d(input_shape_1[3], filter_sizes_1[2], strides_1[3], padding_1, dilations_1[3])
    out_backprop_shape_1 = (input_shape_1[0], out_d_1, out_h_1, out_w_1, filter_sizes_1[4])
    input_dict_1 = {
        'input': np.random.randn(*input_shape_1).astype(np.float32),
        'filter_sizes': filter_sizes_1,
        'out_backprop': np.random.randn(*out_backprop_shape_1).astype(np.float32),
        'strides': strides_1,
        'padding': padding_1,
        'data_format': 'NDHWC',
        'dilations': dilations_1,
        'name': 'test_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: SAME padding, NDHWC
    input_shape_2 = (1, 5, 6, 7, 2)
    filter_sizes_2 = np.array([3, 3, 3, 2, 8], dtype=np.int32)
    strides_2 = [1, 1, 1, 1, 1]
    padding_2 = "SAME"
    dilations_2 = [1, 1, 1, 1, 1]
    out_d_2 = _get_out_size_conv3d(input_shape_2[1], filter_sizes_2[0], strides_2[1], padding_2, dilations_2[1])
    out_h_2 = _get_out_size_conv3d(input_shape_2[2], filter_sizes_2[1], strides_2[2], padding_2, dilations_2[2])
    out_w_2 = _get_out_size_conv3d(input_shape_2[3], filter_sizes_2[2], strides_2[3], padding_2, dilations_2[3])
    out_backprop_shape_2 = (input_shape_2[0], out_d_2, out_h_2, out_w_2, filter_sizes_2[4])
    input_dict_2 = {
        'input': np.random.randn(*input_shape_2).astype(np.float32),
        'filter_sizes': filter_sizes_2,
        'out_backprop': np.random.randn(*out_backprop_shape_2).astype(np.float32),
        'strides': strides_2,
        'padding': padding_2,
        'data_format': 'NDHWC',
        'dilations': dilations_2,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: NCDHW data format
    input_shape_3 = (2, 3, 5, 6, 7) # N, C, D, H, W
    filter_sizes_3 = np.array([3, 3, 3, 3, 4], dtype=np.int32)
    strides_3 = [1, 1, 1, 1, 1]
    padding_3 = "VALID"
    dilations_3 = [1, 1, 1, 1, 1]
    out_d_3 = _get_out_size_conv3d(input_shape_3[2], filter_sizes_3[0], strides_3[2], padding_3, dilations_3[2])
    out_h_3 = _get_out_size_conv3d(input_shape_3[3], filter_sizes_3[1], strides_3[3], padding_3, dilations_3[3])
    out_w_3 = _get_out_size_conv3d(input_shape_3[4], filter_sizes_3[2], strides_3[4], padding_3, dilations_3[4])
    out_backprop_shape_3 = (input_shape_3[0], filter_sizes_3[4], out_d_3, out_h_3, out_w_3)
    input_dict_3 = {
        'input': np.random.randn(*input_shape_3).astype(np.float32),
        'filter_sizes': filter_sizes_3,
        'out_backprop': np.random.randn(*out_backprop_shape_3).astype(np.float32),
        'strides': strides_3,
        'padding': padding_3,
        'data_format': 'NCDHW',
        'dilations': dilations_3,
        'name': 'test_ncdhw'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Strides > 1, float64
    input_shape_4 = (1, 10, 10, 10, 2)
    filter_sizes_4 = np.array([3, 3, 3, 2, 5], dtype=np.int32)
    strides_4 = [1, 2, 2, 2, 1]
    padding_4 = "VALID"
    dilations_4 = [1, 1, 1, 1, 1]
    out_d_4 = _get_out_size_conv3d(input_shape_4[1], filter_sizes_4[0], strides_4[1], padding_4, dilations_4[1])
    out_h_4 = _get_out_size_conv3d(input_shape_4[2], filter_sizes_4[1], strides_4[2], padding_4, dilations_4[2])
    out_w_4 = _get_out_size_conv3d(input_shape_4[3], filter_sizes_4[2], strides_4[3], padding_4, dilations_4[3])
    out_backprop_shape_4 = (input_shape_4[0], out_d_4, out_h_4, out_w_4, filter_sizes_4[4])
    input_dict_4 = {
        'input': np.random.randn(*input_shape_4).astype(np.float64),
        'filter_sizes': filter_sizes_4,
        'out_backprop': np.random.randn(*out_backprop_shape_4).astype(np.float64),
        'strides': strides_4,
        'padding': padding_4,
        'data_format': 'NDHWC',
        'dilations': dilations_4,
        'name': 'test_strides_f64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: float16, previously had invalid dilations
    input_shape_5 = (1, 10, 10, 10, 2)
    filter_sizes_5 = np.array([3, 3, 3, 2, 5], dtype=np.int32)
    strides_5 = [1, 1, 1, 1, 1]
    padding_5 = "VALID"
    dilations_5 = [1, 1, 1, 1, 1] # Fixed dilations
    out_d_5 = _get_out_size_conv3d(input_shape_5[1], filter_sizes_5[0], strides_5[1], padding_5, dilations_5[1])
    out_h_5 = _get_out_size_conv3d(input_shape_5[2], filter_sizes_5[1], strides_5[2], padding_5, dilations_5[2])
    out_w_5 = _get_out_size_conv3d(input_shape_5[3], filter_sizes_5[2], strides_5[3], padding_5, dilations_5[3])
    out_backprop_shape_5 = (input_shape_5[0], out_d_5, out_h_5, out_w_5, filter_sizes_5[4])
    input_dict_5 = {
        'input': np.random.randn(*input_shape_5).astype(np.float16),
        'filter_sizes': filter_sizes_5,
        'out_backprop': np.random.randn(*out_backprop_shape_5).astype(np.float16),
        'strides': strides_5,
        'padding': padding_5,
        'data_format': 'NDHWC',
        'dilations': dilations_5,
        'name': 'test_dilations_f16'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Combined strides, SAME padding, previously had invalid dilations
    input_shape_6 = (2, 12, 12, 12, 4)
    filter_sizes_6 = np.array([3, 3, 3, 4, 8], dtype=np.int32)
    strides_6 = [1, 2, 3, 2, 1]
    padding_6 = "SAME"
    dilations_6 = [1, 1, 1, 1, 1] # Fixed dilations
    out_d_6 = _get_out_size_conv3d(input_shape_6[1], filter_sizes_6[0], strides_6[1], padding_6, dilations_6[1])
    out_h_6 = _get_out_size_conv3d(input_shape_6[2], filter_sizes_6[1], strides_6[2], padding_6, dilations_6[2])
    out_w_6 = _get_out_size_conv3d(input_shape_6[3], filter_sizes_6[2], strides_6[3], padding_6, dilations_6[3])
    out_backprop_shape_6 = (input_shape_6[0], out_d_6, out_h_6, out_w_6, filter_sizes_6[4])
    input_dict_6 = {
        'input': np.random.randn(*input_shape_6).astype(np.float32),
        'filter_sizes': filter_sizes_6,
        'out_backprop': np.random.randn(*out_backprop_shape_6).astype(np.float32),
        'strides': strides_6,
        'padding': padding_6,
        'data_format': 'NDHWC',
        'dilations': dilations_6,
        'name': 'test_combo'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: NCDHW with strides and SAME padding
    input_shape_7 = (1, 3, 7, 8, 9) # N, C, D, H, W
    filter_sizes_7 = np.array([3, 3, 3, 3, 6], dtype=np.int32)
    strides_7 = [1, 1, 2, 2, 2]
    padding_7 = "SAME"
    dilations_7 = [1, 1, 1, 1, 1]
    out_d_7 = _get_out_size_conv3d(input_shape_7[2], filter_sizes_7[0], strides_7[2], padding_7, dilations_7[2])
    out_h_7 = _get_out_size_conv3d(input_shape_7[3], filter_sizes_7[1], strides_7[3], padding_7, dilations_7[3])
    out_w_7 = _get_out_size_conv3d(input_shape_7[4], filter_sizes_7[2], strides_7[4], padding_7, dilations_7[4])
    out_backprop_shape_7 = (input_shape_7[0], filter_sizes_7[4], out_d_7, out_h_7, out_w_7)
    input_dict_7 = {
        'input': np.random.randn(*input_shape_7).astype(np.float32),
        'filter_sizes': filter_sizes_7,
        'out_backprop': np.random.randn(*out_backprop_shape_7).astype(np.float32),
        'strides': strides_7,
        'padding': padding_7,
        'data_format': 'NCDHW',
        'dilations': dilations_7,
        'name': 'test_ncdhw_strides'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Minimal dimensions
    input_shape_8 = (1, 2, 2, 2, 1)
    filter_sizes_8 = np.array([2, 2, 2, 1, 1], dtype=np.int32)
    strides_8 = [1, 1, 1, 1, 1]
    padding_8 = "VALID"
    dilations_8 = [1, 1, 1, 1, 1]
    out_d_8 = _get_out_size_conv3d(input_shape_8[1], filter_sizes_8[0], strides_8[1], padding_8, dilations_8[1])
    out_h_8 = _get_out_size_conv3d(input_shape_8[2], filter_sizes_8[1], strides_8[2], padding_8, dilations_8[2])
    out_w_8 = _get_out_size_conv3d(input_shape_8[3], filter_sizes_8[2], strides_8[3], padding_8, dilations_8[3])
    out_backprop_shape_8 = (input_shape_8[0], out_d_8, out_h_8, out_w_8, filter_sizes_8[4])
    input_dict_8 = {
        'input': np.random.randn(*input_shape_8).astype(np.float32),
        'filter_sizes': filter_sizes_8,
        'out_backprop': np.random.randn(*out_backprop_shape_8).astype(np.float32),
        'strides': strides_8,
        'padding': padding_8,
        'data_format': 'NDHWC',
        'dilations': dilations_8,
        'name': 'test_minimal'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Asymmetric strides, VALID
    input_shape_9 = (1, 10, 10, 10, 1)
    filter_sizes_9 = np.array([2, 2, 2, 1, 2], dtype=np.int32)
    strides_9 = [1, 1, 2, 3, 1]
    padding_9 = "VALID"
    dilations_9 = [1, 1, 1, 1, 1]
    out_d_9 = _get_out_size_conv3d(input_shape_9[1], filter_sizes_9[0], strides_9[1], padding_9, dilations_9[1])
    out_h_9 = _get_out_size_conv3d(input_shape_9[2], filter_sizes_9[1], strides_9[2], padding_9, dilations_9[2])
    out_w_9 = _get_out_size_conv3d(input_shape_9[3], filter_sizes_9[2], strides_9[3], padding_9, dilations_9[3])
    out_backprop_shape_9 = (input_shape_9[0], out_d_9, out_h_9, out_w_9, filter_sizes_9[4])
    input_dict_9 = {
        'input': np.random.randn(*input_shape_9).astype(np.float32),
        'filter_sizes': filter_sizes_9,
        'out_backprop': np.random.randn(*out_backprop_shape_9).astype(np.float32),
        'strides': strides_9,
        'padding': padding_9,
        'data_format': 'NDHWC',
        'dilations': dilations_9,
        'name': 'test_asymmetric_strides'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Asymmetric filter, NCDHW, previously had invalid dilations
    input_shape_10 = (1, 2, 10, 11, 12) # N, C, D, H, W
    filter_sizes_10 = np.array([3, 4, 5, 2, 6], dtype=np.int32)
    strides_10 = [1, 1, 1, 2, 3]
    padding_10 = "SAME"
    dilations_10 = [1, 1, 1, 1, 1] # Fixed dilations
    out_d_10 = _get_out_size_conv3d(input_shape_10[2], filter_sizes_10[0], strides_10[2], padding_10, dilations_10[2])
    out_h_10 = _get_out_size_conv3d(input_shape_10[3], filter_sizes_10[1], strides_10[3], padding_10, dilations_10[3])
    out_w_10 = _get_out_size_conv3d(input_shape_10[4], filter_sizes_10[2], strides_10[4], padding_10, dilations_10[4])
    out_backprop_shape_10 = (input_shape_10[0], filter_sizes_10[4], out_d_10, out_h_10, out_w_10)
    input_dict_10 = {
        'input': np.random.randn(*input_shape_10).astype(np.float32),
        'filter_sizes': filter_sizes_10,
        'out_backprop': np.random.randn(*out_backprop_shape_10).astype(np.float32),
        'strides': strides_10,
        'padding': padding_10,
        'data_format': 'NCDHW',
        'dilations': dilations_10,
        'name': 'test_asymmetric_filter'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Large dimensions
    input_shape_11 = (4, 20, 20, 20, 16)
    filter_sizes_11 = np.array([5, 5, 5, 16, 32], dtype=np.int32)
    strides_11 = [1, 2, 2, 2, 1]
    padding_11 = "SAME"
    dilations_11 = [1, 1, 1, 1, 1]
    out_d_11 = _get_out_size_conv3d(input_shape_11[1], filter_sizes_11[0], strides_11[1], padding_11, dilations_11[1])
    out_h_11 = _get_out_size_conv3d(input_shape_11[2], filter_sizes_11[1], strides_11[2], padding_11, dilations_11[2])
    out_w_11 = _get_out_size_conv3d(input_shape_11[3], filter_sizes_11[2], strides_11[3], padding_11, dilations_11[3])
    out_backprop_shape_11 = (input_shape_11[0], out_d_11, out_h_11, out_w_11, filter_sizes_11[4])
    input_dict_11 = {
        'input': np.random.randn(*input_shape_11).astype(np.float32),
        'filter_sizes': filter_sizes_11,
        'out_backprop': np.random.randn(*out_backprop_shape_11).astype(np.float32),
        'strides': strides_11,
        'padding': padding_11,
        'data_format': 'NDHWC',
        'dilations': dilations_11,
        'name': 'test_large'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    return list_of_inputs

generated_inputs["tf.raw_ops.Conv3DBackpropFilterV2"] = tf_raw_ops_conv3dbackpropfilterv2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Conv3DBackpropFilterV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Conv3DBackpropFilterV2'.")

check_valid('tf.raw_ops.Conv3DBackpropFilterV2', generated_inputs['tf.raw_ops.Conv3DBackpropFilterV2'], lib="tf", suffix=0)
