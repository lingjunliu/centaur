
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_linearoperatoridentity_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        'num_rows': 2,
        'batch_shape': None,
        'dtype': np.float32,
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'assert_proper_shapes': False,
        'name': 'identity_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'num_rows': 3,
        'batch_shape': [2],
        'dtype': np.int32,
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'assert_proper_shapes': False,
        'name': 'identity_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'num_rows': 4,
        'batch_shape': [2, 3],
        'dtype': np.float64,
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'assert_proper_shapes': True,
        'name': 'identity_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'num_rows': 5,
        'batch_shape': [1, 2, 3],
        'dtype': np.complex64,
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'assert_proper_shapes': False,
        'name': 'identity_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 5
    input_dict = {
        'num_rows': 1,
        'batch_shape': [0],
        'dtype': np.float32,
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'assert_proper_shapes': False,
        'name': 'identity_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'num_rows': 10,
        'batch_shape': [5],
        'dtype': np.int64,
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'assert_proper_shapes': True,
        'name': 'identity_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'num_rows': 7,
        'batch_shape': [2, 2],
        'dtype': np.complex128,
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'assert_proper_shapes': False,
        'name': 'identity_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'num_rows': 6,
        'batch_shape': [1],
        'dtype': np.float16,
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'assert_proper_shapes': False,
        'name': 'identity_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'num_rows': 8,
        'batch_shape': [4,1],
        'dtype': np.uint8,
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'assert_proper_shapes': True,
        'name': 'identity_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'num_rows': 9,
        'batch_shape': [1,1,1],
        'dtype': np.int8,
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'assert_proper_shapes': False,
        'name': 'identity_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.LinearOperatorIdentity"] = tf_linalg_linearoperatoridentity_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.LinearOperatorIdentity' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorIdentity'.")

check_valid('tf.linalg.LinearOperatorIdentity', generated_inputs['tf.linalg.LinearOperatorIdentity'], lib="tf", suffix=0)
