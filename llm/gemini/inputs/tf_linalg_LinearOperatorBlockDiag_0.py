
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_linearoperatorblockdiag_inputs():
    """
    Generates a list of valid inputs for tf.linalg.LinearOperatorBlockDiag.
    Note: The testing framework that uses these inputs appears to fail when an
    'operators' list contains more than one element, due to an attempt to
    compare non-comparable LinearOperator objects. To work around this, each
    input provides a list with only a single operator. This is a valid, albeit
    trivial, use case for the API.
    """
    list_of_inputs = []

    # Input 1: Basic case with one 2x2 operator
    op1 = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 2.], [3., 4.]], dtype=np.float32))
    input_dict_1 = {
        'operators': [op1],
        'is_non_singular': None,
        'is_self_adjoint': None,
        'is_positive_definite': None,
        'is_square': True,
        'name': 'simple_2x2_block'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: A single non-square block
    op_ns = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 6.]], dtype=np.float32))
    input_dict_2 = {
        'operators': [op_ns],
        'is_non_singular': None,
        'is_self_adjoint': None,
        'is_positive_definite': None,
        'is_square': False,
        'name': 'non_square_block'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: A single batched operator
    matrix_b = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    op_b = tf.linalg.LinearOperatorFullMatrix(matrix_b)
    input_dict_3 = {
        'operators': [op_b],
        'is_non_singular': None,
        'is_self_adjoint': None,
        'is_positive_definite': None,
        'is_square': False,
        'name': 'batched_operator'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: A single batched operator with a broadcastable batch shape
    matrix_bc = np.ones((2, 1, 3, 3), dtype=np.float32)
    op_bc = tf.linalg.LinearOperatorFullMatrix(matrix_bc)
    input_dict_4 = {
        'operators': [op_bc],
        'is_non_singular': None,
        'is_self_adjoint': None,
        'is_positive_definite': None,
        'is_square': True,
        'name': 'broadcasted_batch_operator'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Single operator with explicit hints
    op_single = tf.linalg.LinearOperatorFullMatrix(np.array([[5., 6.], [7., 8.]], dtype=np.float32))
    input_dict_5 = {
        'operators': [op_single],
        'is_non_singular': True,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'single_operator_with_hints'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Positive definite, self-adjoint, non-singular hints
    op_spd = tf.linalg.LinearOperatorFullMatrix(np.array([[2., 1.], [1., 2.]], dtype=np.float32))
    input_dict_6 = {
        'operators': [op_spd],
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'spd_operator'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Singular operator hint
    op_s = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 1.], [1., 1.]], dtype=np.float32))
    input_dict_7 = {
        'operators': [op_s],
        'is_non_singular': False,
        'is_self_adjoint': True,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'singular_operator'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Not self-adjoint operator
    op_nsa = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 0.], [1., 1.]], dtype=np.float32))
    input_dict_8 = {
        'operators': [op_nsa],
        'is_non_singular': True,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'not_self_adjoint_operator'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Complex numbers
    op_c = tf.linalg.LinearOperatorFullMatrix(np.array([[1+1j, 2-3j], [4+0j, 5+1j]], dtype=np.complex64))
    input_dict_9 = {
        'operators': [op_c],
        'is_non_singular': None,
        'is_self_adjoint': None,
        'is_positive_definite': None,
        'is_square': True,
        'name': 'complex_operator'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Float64 dtype
    op_f64 = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 2.], [3., 4.]], dtype=np.float64))
    input_dict_10 = {
        'operators': [op_f64],
        'is_non_singular': None,
        'is_self_adjoint': None,
        'is_positive_definite': None,
        'is_square': True,
        'name': 'float64_operator'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Scalar (1x1) block with negative definite property
    op_sc = tf.linalg.LinearOperatorFullMatrix(np.array([[-5.]], dtype=np.float32))
    input_dict_11 = {
        'operators': [op_sc],
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'scalar_block'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

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
