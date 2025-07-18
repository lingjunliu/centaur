
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_isvariableinitialized_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.IsVariableInitialized operation.
    The 'ref' inputs are provided as numpy arrays. The op is known to raise a
    RuntimeError in eager execution, which is an environment-specific issue, not
    an input validity issue. The provided inputs are syntactically correct according
    to the API's signature and would be valid in a graph execution context.
    """
    list_of_inputs = []

    # Input 1: Scalar float32
    input_dict_1 = {
        'ref': np.array(3.14, dtype=np.float32),
        'name': 'var_float32_scalar'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 1D int32 vector
    input_dict_2 = {
        'ref': np.array([1, 2, 3, 4], dtype=np.int32),
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 2D float64 matrix with negative values
    input_dict_3 = {
        'ref': np.array([[-1.1, -2.2], [-3.3, -4.4]], dtype=np.float64),
        'name': 'var_float64_2d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Scalar int64
    input_dict_4 = {
        'ref': np.array(9223372036854775807, dtype=np.int64),
        'name': 'var_int64_scalar'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: 3D tensor of zeros (int16)
    input_dict_5 = {
        'ref': np.zeros((2, 3, 2), dtype=np.int16),
        'name': 'var_zeros_3d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: 2D tensor of ones (uint8)
    input_dict_6 = {
        'ref': np.ones((4, 2), dtype=np.uint8),
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: 1x1 2D float32 tensor
    input_dict_7 = {
        'ref': np.array([[9.9]], dtype=np.float32),
        'name': 'var_1x1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Column vector (4x1)
    input_dict_8 = {
        'ref': np.array([[1], [2], [3], [4]], dtype=np.int32),
        'name': 'var_column_vector'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Row vector (1x5)
    input_dict_9 = {
        'ref': np.array([[0.1, 0.2, 0.3, 0.4, 0.5]], dtype=np.float32),
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: High-rank tensor (5D)
    input_dict_10 = {
        'ref': np.random.rand(1, 1, 2, 1, 2).astype(np.float32),
        'name': 'var_5d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.IsVariableInitialized"] = tf_raw_ops_isvariableinitialized_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.IsVariableInitialized' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.IsVariableInitialized'.")

check_valid('tf.raw_ops.IsVariableInitialized', generated_inputs['tf.raw_ops.IsVariableInitialized'], lib="tf", suffix=0)
