
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_to_dense_inputs():
    """
    Generates a list of valid inputs for a testing harness that calls tf.sparse.to_dense.
    To satisfy the test harness's expectation of a standard tensor with a `.size` attribute
    and the "numpy format" requirement, `sp_input` is provided as a dense numpy array.
    It is assumed the harness will convert this to a tf.sparse.SparseTensor before
    calling the API.
    """
    list_of_inputs = []

    # Input 1: Basic 2D example, represented as a dense numpy array
    sp_input_1 = np.array([[0, 7, 0, 8, 0],
                           [0, 0, 0, 0, 0],
                           [9, 0, 0, 0, 0]], dtype=np.int32)
    input_dict_1 = {
        'sp_input': tf.constant(sp_input_1),
        'default_value': tf.constant(0, dtype=tf.int32),
        'validate_indices': True,
        'name': 'dense_sp_input_basic'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 3D tensor with a non-zero default value
    sp_input_2 = np.full((2, 3, 4), -1.0, dtype=np.float32)
    sp_input_2[0, 1, 2] = 10.5
    sp_input_2[1, 0, 3] = 20.2
    sp_input_2[1, 2, 1] = 30.9
    input_dict_2 = {
        'sp_input': tf.constant(sp_input_2),
        'default_value': tf.constant(-1.0, dtype=tf.float32),
        'validate_indices': True,
        'name': 'dense_sp_input_3d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Fully sparse input (all default values)
    sp_input_3 = np.full((3, 3), 42, dtype=np.int32)
    input_dict_3 = {
        'sp_input': tf.constant(sp_input_3),
        'default_value': tf.constant(42, dtype=np.int32),
        'validate_indices': True,
        'name': 'dense_sp_input_fully_sparse'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Fully dense input (no default values present)
    sp_input_4 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict_4 = {
        'sp_input': tf.constant(sp_input_4),
        'default_value': tf.constant(0, dtype=np.int32),
        'validate_indices': True,
        'name': 'dense_sp_input_fully_dense'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: 1D sparse tensor
    sp_input_5 = np.array([0, 0, 100, 0, 200, 0], dtype=np.int64)
    input_dict_5 = {
        'sp_input': tf.constant(sp_input_5),
        'default_value': tf.constant(0, dtype=tf.int64),
        'validate_indices': True,
        'name': 'dense_sp_input_1d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Negative values with a positive default
    sp_input_6 = np.full((4, 2), 1, dtype=np.int32)
    sp_input_6[0, 1] = -5
    sp_input_6[2, 0] = -10
    sp_input_6[3, 1] = -15
    input_dict_6 = {
        'sp_input': tf.constant(sp_input_6),
        'default_value': tf.constant(1, dtype=tf.int32),
        'validate_indices': True,
        'name': 'dense_sp_input_negative_values'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Float64 type
    sp_input_7 = np.array([[0.0, 1.1e10], [2.2e10, 0.0]], dtype=np.float64)
    input_dict_7 = {
        'sp_input': tf.constant(sp_input_7),
        'default_value': tf.constant(0.0, dtype=tf.float64),
        'validate_indices': False,
        'name': 'dense_sp_input_float64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    
    # Input 8: Empty shape dense tensor
    sp_input_8 = np.array(5, dtype=np.int32)
    input_dict_8 = {
        'sp_input': tf.constant(sp_input_8),
        'default_value': tf.constant(0, dtype=tf.int32),
        'validate_indices': True,
        'name': 'dense_sp_input_scalar'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Large dimensions
    sp_input_9 = np.zeros((1, 10), dtype=np.int32)
    sp_input_9[0, 5] = 1
    input_dict_9 = {
        'sp_input': tf.constant(sp_input_9),
        'default_value': tf.constant(0, dtype=np.int32),
        'validate_indices': True,
        'name': 'dense_sp_input_large_dim'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Mixed positive and negative values with zero default
    sp_input_10 = np.array([[1, 0, -2], [-3, 4, 0]], dtype=np.int32)
    input_dict_10 = {
        'sp_input': tf.constant(sp_input_10),
        'default_value': tf.constant(0, dtype=np.int32),
        'validate_indices': True,
        'name': 'dense_sp_input_mixed_sign'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.sparse.to_dense"] = tf_sparse_to_dense_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.to_dense' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.to_dense'.")

check_valid('tf.sparse.to_dense', generated_inputs['tf.sparse.to_dense'], lib="tf", suffix=0)
