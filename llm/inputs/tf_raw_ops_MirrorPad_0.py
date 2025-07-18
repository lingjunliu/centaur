
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_mirror_pad_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.MirrorPad.
    """
    list_of_inputs = []

    # Input 1: Basic 1D SYMMETRIC padding
    input_dict_1 = {
        'input': np.array([1, 2, 3], dtype=np.int32),
        'paddings': np.array([[2, 2]], dtype=np.int32),
        'mode': 'SYMMETRIC',
        'name': 'test_1d_symmetric'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Basic 1D REFLECT padding
    input_dict_2 = {
        'input': np.array([1, 2, 3], dtype=np.int32),
        'paddings': np.array([[2, 2]], dtype=np.int32),
        'mode': 'REFLECT',
        'name': 'test_1d_reflect'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 2D padding from documentation example (SYMMETRIC)
    input_dict_3 = {
        'input': np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
        'paddings': np.array([[1, 1], [2, 2]], dtype=np.int32),
        'mode': 'SYMMETRIC',
        'name': 'test_2d_symmetric_doc'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 2D padding with float data and REFLECT mode
    input_dict_4 = {
        'input': np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float32),
        'paddings': np.array([[1, 0], [0, 1]], dtype=np.int64),
        'mode': 'REFLECT',
        'name': 'test_2d_reflect_float'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: 3D padding (SYMMETRIC), padding only on some dimensions
    input_dict_5 = {
        'input': np.arange(1, 9).reshape((2, 2, 2)).astype(np.int32),
        'paddings': np.array([[1, 1], [0, 0], [2, 2]], dtype=np.int32),
        'mode': 'SYMMETRIC',
        'name': 'test_3d_symmetric_partial'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: 3D padding (REFLECT) with float64 data
    input_dict_6 = {
        'input': np.random.rand(2, 3, 2).astype(np.float64),
        'paddings': np.array([[1, 1], [2, 2], [1, 1]], dtype=np.int32),
        'mode': 'REFLECT',
        'name': 'test_3d_reflect_float64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: No padding (paddings are all zero)
    input_dict_7 = {
        'input': np.array([[1, 2], [3, 4]], dtype=np.int32),
        'paddings': np.array([[0, 0], [0, 0]], dtype=np.int32),
        'mode': 'SYMMETRIC',
        'name': 'test_no_padding'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: One-sided padding (FIXED)
    # For REFLECT, pad <= dim_size - 1. Here dim_size=3, so pad <= 2.
    input_dict_8 = {
        'input': np.array([10, 20, 30], dtype=np.int64),
        'paddings': np.array([[2, 0]], dtype=np.int32),
        'mode': 'REFLECT',
        'name': 'test_one_sided_padding'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Maximum allowed padding for SYMMETRIC mode
    input_tensor_9 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    # For SYMMETRIC, pad <= dim_size. Here dim_size is 2 for both dims.
    input_dict_9 = {
        'input': input_tensor_9,
        'paddings': np.array([[2, 2], [1, 2]], dtype=np.int32),
        'mode': 'SYMMETRIC',
        'name': 'test_max_padding_symmetric'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Maximum allowed padding for REFLECT mode
    input_tensor_10 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    # For REFLECT, pad <= dim_size - 1.
    # Dim 0: size=2, pad <= 1. Dim 1: size=3, pad <= 2.
    input_dict_10 = {
        'input': input_tensor_10,
        'paddings': np.array([[1, 1], [2, 2]], dtype=np.int32),
        'mode': 'REFLECT',
        'name': 'test_max_padding_reflect'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: 4D tensor padding
    input_dict_11 = {
        'input': np.ones((1, 2, 1, 3), dtype=np.int32),
        'paddings': np.array([[0, 1], [1, 1], [1, 0], [2, 1]], dtype=np.int32),
        'mode': 'SYMMETRIC',
        'name': 'test_4d_symmetric'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    # Input 12: A tensor with a dimension of size 1 (FIXED)
    # For REFLECT mode on a dim of size 1, padding must be 0 (pad <= 1 - 1).
    input_dict_12 = {
        'input': np.array([[[10], [20]]], dtype=np.int32), # shape (1, 2, 1)
        'paddings': np.array([[0, 0], [1, 1], [0, 0]], dtype=np.int32),
        'mode': 'REFLECT',
        'name': 'test_dim_size_one_reflect'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_12))

    return list_of_inputs

generated_inputs["tf.raw_ops.MirrorPad"] = tf_raw_ops_mirror_pad_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.MirrorPad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MirrorPad'.")

check_valid('tf.raw_ops.MirrorPad', generated_inputs['tf.raw_ops.MirrorPad'], lib="tf", suffix=0)
