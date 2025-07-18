
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_linearoperatorblockdiag_inputs():
    """
    Generates a list of valid inputs for tf.linalg.LinearOperatorBlockDiag.
    Note: The 'operators' list contains a single operator to avoid comparison errors
    in testing frameworks that may not handle lists of non-comparable objects.
    """
    list_of_inputs = []

    # Input 1: Basic case, 2x2 operator
    op1 = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 2.], [3., 4.]], dtype=np.float32))
    input_dict_1 = {
        'operators': [op1],
        'is_non_singular': None,
        'is_self_adjoint': None,
        'is_positive_definite': None,
        'is_square': True,
        'name': 'basic_2x2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 3x3 identity operator
    op2 = tf.linalg.LinearOperatorIdentity(num_rows=3, dtype=np.float32)
    input_dict_2 = {
        'operators': [op2],
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'identity_3x3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Non-square operator
    op3 = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32))
    input_dict_3 = {
        'operators': [op3],
        'is_non_singular': None,
        'is_self_adjoint': None,
        'is_positive_definite': None,
        'is_square': False,
        'name': 'non_square'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: A 4x4 diagonal operator
    op4 = tf.linalg.LinearOperatorDiag(diag=np.array([1., -2., 3., -4.], dtype=np.float32))
    input_dict_4 = {
        'operators': [op4],
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'diag_operator'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Batch of 3x3 operators
    op5 = tf.linalg.LinearOperatorFullMatrix(np.random.rand(4, 3, 3).astype(np.float32))
    input_dict_5 = {
        'operators': [op5],
        'is_non_singular': None,
        'is_self_adjoint': None,
        'is_positive_definite': None,
        'is_square': True,
        'name': 'batch_operator'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Batch of non-square operators
    op6 = tf.linalg.LinearOperatorFullMatrix(np.random.rand(2, 4, 2).astype(np.float32))
    input_dict_6 = {
        'operators': [op6],
        'is_non_singular': None,
        'is_self_adjoint': None,
        'is_positive_definite': None,
        'is_square': False,
        'name': 'batch_non_square'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Using True hints (for a positive-definite operator)
    op7_matrix = np.array([[2., 1.], [1., 2.]], dtype=np.float32)
    op7 = tf.linalg.LinearOperatorFullMatrix(op7_matrix)
    input_dict_7 = {
        'operators': [op7],
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'true_hints'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Using False hints (for a singular operator)
    op8_matrix = np.array([[1., 1.], [1., 1.]], dtype=np.float32)
    op8 = tf.linalg.LinearOperatorFullMatrix(op8_matrix)
    input_dict_8 = {
        'operators': [op8],
        'is_non_singular': False,
        'is_self_adjoint': True,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'false_hints'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: float64 dtype
    op9 = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 2.], [3., 4.]], dtype=np.float64))
    input_dict_9 = {
        'operators': [op9],
        'is_non_singular': None,
        'is_self_adjoint': None,
        'is_positive_definite': None,
        'is_square': True,
        'name': 'float64_dtype'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: complex64 dtype
    op10_matrix = np.array([[1.+1.j, 2.+2.j], [3.+3.j, 4.+4.j]], dtype=np.complex64)
    op10 = tf.linalg.LinearOperatorFullMatrix(op10_matrix)
    input_dict_10 = {
        'operators': [op10],
        'is_non_singular': None,
        'is_self_adjoint': None,
        'is_positive_definite': None,
        'is_square': True,
        'name': 'complex64_dtype'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Zeros operator (non-square, created with FullMatrix to be safe)
    op11 = tf.linalg.LinearOperatorFullMatrix(np.zeros((5, 4), dtype=np.float32))
    input_dict_11 = {
        'operators': [op11],
        'is_non_singular': False,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': False,
        'name': 'zeros_operator_non_square'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    # Input 12: Scaled identity operator with batch shape
    op12 = tf.linalg.LinearOperatorScaledIdentity(
        num_rows=3,
        multiplier=tf.constant([1., 2., 3.], dtype=np.float32),
        is_self_adjoint=True,
        is_positive_definite=True)
    input_dict_12 = {
        'operators': [op12],
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'batch_scaled_identity'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_12))

    return list_of_inputs

generated_inputs["tf.linalg.LinearOperatorBlockDiag"] = tf_linalg_linearoperatorblockdiag_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.linalg.LinearOperatorBlockDiag' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorBlockDiag'.")

check_valid('tf.linalg.LinearOperatorBlockDiag', generated_inputs['tf.linalg.LinearOperatorBlockDiag'], lib="tf", suffix=0)
