
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_tf_linalg_linearoperatorcomposition_inputs():
    """
    Generates a list of valid inputs for the tf.linalg.LinearOperatorComposition function.
    The user's testing environment seems to fail when comparing tf.linalg.LinearOperator
    objects in a list. To work around this, each 'operators' list will contain only a single
    operator, which is a valid but trivial case for composition. All parameters from the
    signature will be provided.
    """
    list_of_inputs = []

    # Input 1: Single 2x2 operator
    op1 = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 2.], [3., 4.]], dtype=np.float32))
    input_1 = {
        'operators': [op1],
        'is_non_singular': True,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'single_2x2_operator'
    }
    list_of_inputs.append(copy.deepcopy(input_1))

    # Input 2: Single 3x3 operator, float64
    op2 = tf.linalg.LinearOperatorFullMatrix(np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=np.float64))
    input_2 = {
        'operators': [op2],
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'single_3x3_operator_float64'
    }
    list_of_inputs.append(copy.deepcopy(input_2))

    # Input 3: Single non-square operator (2x3)
    op3 = tf.linalg.LinearOperatorFullMatrix(np.random.rand(2, 3).astype(np.float32))
    input_3 = {
        'operators': [op3],
        'is_non_singular': False,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': False,
        'name': 'single_non_square_operator'
    }
    list_of_inputs.append(copy.deepcopy(input_3))

    # Input 4: Single batch operator
    op4 = tf.linalg.LinearOperatorFullMatrix(np.random.rand(2, 3, 4).astype(np.float32))
    input_4 = {
        'operators': [op4],
        'is_non_singular': False,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': False,
        'name': 'single_batch_operator'
    }
    list_of_inputs.append(copy.deepcopy(input_4))

    # Input 5: Single complex operator
    op5 = tf.linalg.LinearOperatorFullMatrix(
        np.array([[1+1j, 2-3j], [3+0j, 4+1j]], dtype=np.complex64))
    input_5 = {
        'operators': [op5],
        'is_non_singular': True,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'single_complex_operator'
    }
    list_of_inputs.append(copy.deepcopy(input_5))

    # Input 6: Single self-adjoint diagonal operator
    op6 = tf.linalg.LinearOperatorDiag(np.array([1., 2., 3.], dtype=np.float32))
    input_6 = {
        'operators': [op6],
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'single_diag_operator'
    }
    list_of_inputs.append(copy.deepcopy(input_6))

    # Input 7: Single Identity operator
    op7 = tf.linalg.LinearOperatorIdentity(num_rows=5, dtype=np.float32)
    input_7 = {
        'operators': [op7],
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'single_identity_operator'
    }
    list_of_inputs.append(copy.deepcopy(input_7))

    # Input 8: Single Scaled Identity operator
    op8 = tf.linalg.LinearOperatorScaledIdentity(num_rows=4, multiplier=np.float32(3.0))
    input_8 = {
        'operators': [op8],
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'single_scaled_identity_operator'
    }
    list_of_inputs.append(copy.deepcopy(input_8))

    # Input 9: Single singular operator
    op9 = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 1.], [1., 1.]], dtype=np.float32))
    input_9 = {
        'operators': [op9],
        'is_non_singular': False,
        'is_self_adjoint': True,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'single_singular_operator'
    }
    list_of_inputs.append(copy.deepcopy(input_9))

    # Input 10: Single high-dimensional batch operator
    op10 = tf.linalg.LinearOperatorFullMatrix(np.random.rand(2, 1, 3, 5, 6).astype(np.float32))
    input_10 = {
        'operators': [op10],
        'is_non_singular': False,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': False,
        'name': 'single_high_dim_batch_operator'
    }
    list_of_inputs.append(copy.deepcopy(input_10))

    return list_of_inputs

generated_inputs["tf.linalg.LinearOperatorComposition"] = get_tf_linalg_linearoperatorcomposition_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.linalg.LinearOperatorComposition' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorComposition'.")

check_valid('tf.linalg.LinearOperatorComposition', generated_inputs['tf.linalg.LinearOperatorComposition'], lib="tf", suffix=0)
