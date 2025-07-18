
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_debug_gradient_ref_identity_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.DebugGradientRefIdentity function.
    The op itself is designed for graph mode and will raise a RuntimeError in eager execution.
    The provided inputs are valid data structures that will be converted to tensors.
    """
    list_of_inputs = []

    # Input 1: Basic float32 scalar
    input_dict_1 = {
        'input': np.array(3.14, dtype=np.float32),
        'name': 'f32_scalar'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Basic int32 scalar
    input_dict_2 = {
        'input': np.array(42, dtype=np.int32),
        'name': 'i32_scalar'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 1D float32 vector
    input_dict_3 = {
        'input': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        'name': 'f32_vector'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 1D int32 vector with negative values
    input_dict_4 = {
        'input': np.array([-1, 0, 1], dtype=np.int32),
        'name': 'i32_vector'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: 2D float32 matrix
    input_dict_5 = {
        'input': np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float32),
        'name': 'f32_matrix'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: 2D int32 matrix (identity)
    input_dict_6 = {
        'input': np.eye(3, dtype=np.int32),
        'name': 'i32_matrix'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: 3D float32 tensor of zeros
    input_dict_7 = {
        'input': np.zeros((2, 2, 2), dtype=np.float32),
        'name': 'f32_3d_zeros'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: 3D int32 tensor of ones
    input_dict_8 = {
        'input': np.ones((1, 2, 3), dtype=np.int32),
        'name': 'i32_3d_ones'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: float64 column vector
    input_dict_9 = {
        'input': np.array([[1.0], [2.0], [3.0]], dtype=np.float64),
        'name': 'f64_col_vector'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: int64 row vector
    input_dict_10 = {
        'input': np.array([[100, 200, 300]], dtype=np.int64),
        'name': 'i64_row_vector'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.DebugGradientRefIdentity"] = tf_raw_ops_debug_gradient_ref_identity_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.DebugGradientRefIdentity' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DebugGradientRefIdentity'.")

check_valid('tf.raw_ops.DebugGradientRefIdentity', generated_inputs['tf.raw_ops.DebugGradientRefIdentity'], lib="tf", suffix=0)
