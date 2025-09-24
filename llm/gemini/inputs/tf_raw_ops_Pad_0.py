
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_pad_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.Pad function.
    """
    list_of_inputs = []

    # Input 1: Basic 2D case from the documentation
    input_dict_1 = {
        'name': 'doc_example',
        'input': np.array([[1, 1], [2, 2]], dtype=np.int32),
        'paddings': np.array([[1, 1], [2, 2]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 1D tensor padding
    input_dict_2 = {
        'name': 'pad_1d_tensor',
        'input': np.array([10, 20, 30], dtype=np.int32),
        'paddings': np.array([[2, 3]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 3D tensor padding with varied padding on each dimension
    input_dict_3 = {
        'name': 'pad_3d_tensor',
        'input': np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32),
        'paddings': np.array([[1, 0], [0, 1], [2, 1]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: No padding (paddings are all zeros)
    input_dict_4 = {
        'name': 'no_padding',
        'input': np.array([[1, 2], [3, 4]], dtype=np.int32),
        'paddings': np.array([[0, 0], [0, 0]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Padding on one side only (before)
    input_dict_5 = {
        'name': 'pad_before_only',
        'input': np.array([5, 6, 7], dtype=np.int32),
        'paddings': np.array([[4, 0]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Padding on one side only (after)
    input_dict_6 = {
        'name': 'pad_after_only',
        'input': np.array([[10], [20]], dtype=np.int32),
        'paddings': np.array([[0, 2], [0, 3]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Using int64 for the paddings tensor
    input_dict_7 = {
        'name': 'int64_paddings',
        'input': np.array([1, 2, 3], dtype=np.int32),
        'paddings': np.array([[1, 2]], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Higher rank tensor (4D)
    input_dict_8 = {
        'name': 'pad_4d_tensor',
        'input': np.ones((1, 2, 1, 3), dtype=np.int8),
        'paddings': np.array([[0, 1], [1, 0], [2, 1], [0, 0]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Scalar input (rank 0)
    input_dict_9 = {
        'name': 'scalar_input_no_padding',
        'input': np.array(100, dtype=np.int32),
        'paddings': np.array([], dtype=np.int32).reshape(0, 2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Input tensor with a zero dimension
    input_dict_10 = {
        'name': 'zero_dimension_input',
        'input': np.empty((2, 0, 3), dtype=np.float64),
        'paddings': np.array([[1, 1], [2, 2], [0, 1]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Large padding values
    input_dict_11 = {
        'name': 'large_paddings',
        'input': np.array([1], dtype=np.int32),
        'paddings': np.array([[10, 20]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))
    
    # Input 12: Empty input tensor
    input_dict_12 = {
        'name': 'empty_input',
        'input': np.array([], dtype=np.int32),
        'paddings': np.array([[3, 4]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_12))

    return list_of_inputs

generated_inputs["tf.raw_ops.Pad"] = tf_raw_ops_pad_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Pad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Pad'.")

check_valid('tf.raw_ops.Pad', generated_inputs['tf.raw_ops.Pad'], lib="tf", suffix=0)
