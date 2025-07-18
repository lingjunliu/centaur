
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_destroytemporaryvariable_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.DestroyTemporaryVariable.
    NOTE: This operation is fundamentally incompatible with eager execution, which is
    the default mode in TensorFlow 2.x. It is a legacy op designed for graph mode
    and expects a 'ref' type tensor, which cannot be created from a NumPy array
    in an eager context. Therefore, calling this function in an eager environment
    will always raise a `RuntimeError`, as observed in the traceback. The provided
    inputs are valid with respect to the API's signature (data types and parameter names)
    but are guaranteed to fail at runtime in the testing environment.
    """
    list_of_inputs = []

    # Input 1: Basic float32 vector.
    input_dict_1 = {
        'ref': np.array([1.0, 2.5, -3.0], dtype=np.float32),
        'var_name': 'temp_var_float',
        'name': 'destroy_op_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D int32 matrix.
    input_dict_2 = {
        'ref': np.array([[-1, 0], [100, -200]], dtype=np.int32),
        'var_name': 'my_temp_int_matrix',
        'name': 'destroy_op_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Scalar (0-D) float64.
    input_dict_3 = {
        'ref': np.array(3.14159, dtype=np.float64),
        'var_name': 'scalar_var',
        'name': 'destroy_op_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Boolean tensor.
    input_dict_4 = {
        'ref': np.array([[True, False], [False, True]], dtype=np.bool_),
        'var_name': 'bool_temp_var',
        'name': 'destroy_op_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Complex number tensor (complex64).
    input_dict_5 = {
        'ref': np.array([1+2j, 3-4j], dtype=np.complex64),
        'var_name': 'complex_var_64',
        'name': 'destroy_op_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    return list_of_inputs

generated_inputs["tf.raw_ops.DestroyTemporaryVariable"] = tf_raw_ops_destroytemporaryvariable_inputs()

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
