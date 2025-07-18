
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_linearoperatorinversion_inputs():
    list_of_inputs = []

    # Helper function to create operator and add .size attribute
    def create_operator(matrix):
        op = tf.linalg.LinearOperatorFullMatrix(matrix)
        op.size = np.size(matrix)
        return op

    # Input 1: Simple 2x2 Identity matrix
    op1 = create_operator(np.array([[1., 0.], [0., 1.]], dtype=np.float32))
    input_dict_1 = {
        'operator': op1,
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'identity_inversion'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2x2 Diagonal matrix with float values
    op2 = create_operator(np.array([[2., 0.], [0., 5.]], dtype=np.float32))
    input_dict_2 = {
        'operator': op2,
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'diagonal_inversion'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 2x2 Symmetric but not positive-definite
    op3 = create_operator(np.array([[1., 2.], [2., 1.]], dtype=np.float32))
    input_dict_3 = {
        'operator': op3,
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'symmetric_indefinite_inversion'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 3x3 non-singular matrix, some hints are None
    op4 = create_operator(np.array([[1., 2., 3.], [0., 1., 4.], [5., 6., 0.]], dtype=np.float32))
    input_dict_4 = {
        'operator': op4,
        'is_non_singular': True,
        'is_self_adjoint': None,
        'is_positive_definite': None,
        'is_square': True,
        'name': 'general_3x3_inversion'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Batch of 2x2 matrices
    op5 = create_operator(np.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]], dtype=np.float32))
    input_dict_5 = {
        'operator': op5,
        'is_non_singular': True,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'batch_2x2_inversion'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Diagonal matrix with negative values
    op6 = create_operator(np.array([[-1., 0.], [0., -2.]], dtype=np.float32))
    input_dict_6 = {
        'operator': op6,
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'negative_definite_inversion'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Different data type (float64)
    op7 = create_operator(np.array([[10., 1.], [1., 10.]], dtype=np.float64))
    input_dict_7 = {
        'operator': op7,
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'float64_inversion'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: All hints are None (default behavior)
    op8 = create_operator(np.array([[3., -1.], [-1., 3.]], dtype=np.float32))
    input_dict_8 = {
        'operator': op8,
        'is_non_singular': None,
        'is_self_adjoint': None,
        'is_positive_definite': None,
        'is_square': None,
        'name': 'no_hints_inversion'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Larger 4x4 matrix
    op9 = create_operator(np.eye(4, dtype=np.float32) + np.diag(np.ones(3, dtype=np.float32), k=1))
    input_dict_9 = {
        'operator': op9,
        'is_non_singular': True,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'large_4x4_inversion'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Batch of 3x3 matrices where batch members have different properties
    op10 = create_operator(np.array([
        [[2., 1., 0.], [1., 2., 1.], [0., 1., 2.]], 
        [[1., 2., 3.], [4., 5., 6.], [7., 8., 10.]]
    ], dtype=np.float32))
    input_dict_10 = {
        'operator': op10,
        'is_non_singular': True,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'batch_mixed_property_inversion'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))
    
    return list_of_inputs

generated_inputs["tf.linalg.LinearOperatorInversion"] = tf_linalg_linearoperatorinversion_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.linalg.LinearOperatorInversion' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorInversion'.")

check_valid('tf.linalg.LinearOperatorInversion', generated_inputs['tf.linalg.LinearOperatorInversion'], lib="tf", suffix=0)
