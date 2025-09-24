
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy


def get_destroy_temporary_variable_inputs():
    """
    Generates a list of inputs for the tf.raw_ops.DestroyTemporaryVariable function.
    NOTE: This operation is designed for TensorFlow's graph mode and is expected to fail
    when called directly in eager execution. The 'ref' argument must be a reference
    to a TemporaryVariable created within the same graph. The provided numpy arrays
    are placeholders that conform to the required signature but will cause a
    runtime error in the eager execution context of the test environment.
    """
    list_of_inputs = []

    # Input 1: Basic float32 1D tensor
    input_dict_1 = {
        'ref': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        'var_name': 'temp_var_float32',
        'name': 'destroy_op_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: int32 2D tensor with negative values
    input_dict_2 = {
        'ref': np.array([[-1, 2], [3, -4]], dtype=np.int32),
        'var_name': 'temp_var_int32_matrix',
        'name': 'DestroyIntMatrix'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Scalar (0D) float64 tensor
    input_dict_3 = {
        'ref': np.array(42.0, dtype=np.float64),
        'var_name': 'scalar_var',
        'name': 'destroy_scalar'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Empty 1D tensor
    input_dict_4 = {
        'ref': np.array([], dtype=np.float32),
        'var_name': 'empty_variable',
        'name': 'destroy_empty'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Tensor with a zero dimension
    input_dict_5 = {
        'ref': np.zeros((3, 0, 2), dtype=np.int64),
        'var_name': 'zero_dim_var',
        'name': 'destroy_zero_dim_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Boolean 3D tensor
    input_dict_6 = {
        'ref': np.array([[[True], [False]], [[False], [True]]], dtype=np.bool_),
        'var_name': 'boolean_tensor_var',
        'name': 'destroy_bools'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: uint8 tensor
    input_dict_7 = {
        'ref': np.array([0, 127, 255], dtype=np.uint8),
        'var_name': 'image_data_like',
        'name': 'destroy_uint8_data'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Large 2D tensor
    input_dict_8 = {
        'ref': np.arange(100, dtype=np.float32).reshape(10, 10),
        'var_name': 'large_matrix_var',
        'name': 'destroy_large_matrix'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: High-rank (5D) tensor with float32
    input_dict_9 = {
        'ref': np.ones((1, 2, 1, 3, 1), dtype=np.float32),
        'var_name': 'high_rank_variable',
        'name': 'destroy_5D'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: A different 1D tensor
    input_dict_10 = {
        'ref': np.array([5.5, 6.6, 7.7, 8.8], dtype=np.float32),
        'var_name': 'another_float_var',
        'name': 'destroy_op_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.DestroyTemporaryVariable"] = get_destroy_temporary_variable_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.DestroyTemporaryVariable' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DestroyTemporaryVariable'.")

check_valid('tf.raw_ops.DestroyTemporaryVariable', generated_inputs['tf.raw_ops.DestroyTemporaryVariable'], lib="tf", suffix=0)
