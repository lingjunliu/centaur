
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

def get_temporary_variable_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.TemporaryVariable function.
    """
    list_of_inputs = []

    # Input 1: Basic 1D float32 variable
    input_dict = {
        'shape': [10],
        'dtype': np.float32,
        'var_name': '',
        'name': 'basic_float_var'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int32 variable with a custom var_name
    input_dict = {
        'shape': [3, 4],
        'dtype': np.int32,
        'var_name': 'my_temp_var_int',
        'name': 'int_matrix_var'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float64 variable
    input_dict = {
        'shape': [2, 3, 5],
        'dtype': np.float64,
        'var_name': '',
        'name': 'float64_tensor_var'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Scalar (0-D) int64 variable
    input_dict = {
        'shape': [],
        'dtype': np.int64,
        'var_name': 'scalar_var',
        'name': 'scalar_op'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D bool variable
    input_dict = {
        'shape': [100],
        'dtype': np.bool_,
        'var_name': '',
        'name': 'bool_vector_var'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D complex64 variable with custom names
    input_dict = {
        'shape': [2, 2, 2],
        'dtype': np.complex64,
        'var_name': 'complex_cube',
        'name': 'complex64_op'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D complex128 variable
    input_dict = {
        'shape': [5, 5],
        'dtype': np.complex128,
        'var_name': '',
        'name': 'complex128_matrix'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Variable with a dimension of size 1
    input_dict = {
        'shape': [1, 10, 1],
        'dtype': np.int16,
        'var_name': 'singleton_dim_var',
        'name': 'op_with_singleton'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Variable with a zero-sized dimension
    input_dict = {
        'shape': [10, 0, 5],
        'dtype': np.float32,
        'var_name': 'zero_dim_var',
        'name': 'op_with_zero'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Unsigned integer type uint8
    input_dict = {
        'shape': [8, 8],
        'dtype': np.uint8,
        'var_name': 'uint8_var',
        'name': 'unsigned_int_op'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.TemporaryVariable"] = get_temporary_variable_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.TemporaryVariable' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.TemporaryVariable'.")

check_valid('tf.raw_ops.TemporaryVariable', generated_inputs['tf.raw_ops.TemporaryVariable'], lib="tf", suffix=0)
