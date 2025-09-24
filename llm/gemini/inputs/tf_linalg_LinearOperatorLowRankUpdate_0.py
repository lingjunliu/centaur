
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf

def tf_linalg_linearoperatorlowrankupdate_inputs():
    """
    Returns a list of valid inputs for tf.linalg.LinearOperatorLowRankUpdate.
    """
    list_of_inputs = []

    # Input 1: Basic square case (M=N=3, K=2), all hints True
    base_matrix_1 = np.identity(3, dtype=np.float32)
    base_op_1 = tf.linalg.LinearOperatorFullMatrix(
        matrix=base_matrix_1,
        is_non_singular=True, is_self_adjoint=True,
        is_positive_definite=True, is_square=True)
    base_op_1.size = base_matrix_1.size
    input_dict_1 = {
        'base_operator': base_op_1,
        'u': np.array([[1., 2.], [-1., 3.], [0., 1.]], dtype=np.float32),
        'diag_update': np.array([11., 12.], dtype=np.float32),
        'v': np.array([[1., 2.], [-1., 3.], [1., 1.]], dtype=np.float32),
        'is_diag_update_positive': True,
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'simple_square_pd'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Non-self-adjoint case with rank-1 update
    base_matrix_2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    base_op_2 = tf.linalg.LinearOperatorFullMatrix(
        matrix=base_matrix_2,
        is_non_singular=False, is_self_adjoint=False,
        is_positive_definite=False, is_square=True)
    base_op_2.size = base_matrix_2.size
    input_dict_2 = {
        'base_operator': base_op_2,
        'u': np.array([[1.], [2.], [3.]], dtype=np.float32),
        'diag_update': np.array([5.], dtype=np.float32),
        'v': np.array([[9.], [8.], [7.]], dtype=np.float32),
        'is_diag_update_positive': None,
        'is_non_singular': False,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'non_self_adjoint_rank1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: `v` is None, implying v=u (symmetric update)
    base_matrix_3 = np.diag([1., 2., 3.]).astype(np.float32)
    base_op_3 = tf.linalg.LinearOperatorFullMatrix(
        matrix=base_matrix_3,
        is_non_singular=True, is_self_adjoint=True,
        is_positive_definite=True, is_square=True)
    base_op_3.size = base_matrix_3.size
    input_dict_3 = {
        'base_operator': base_op_3,
        'u': np.array([[1., 2.], [3., 4.], [5., 6.]], dtype=np.float32),
        'diag_update': np.array([-1., 1.], dtype=np.float32),
        'v': None,
        'is_diag_update_positive': False,
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'v_is_none'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: `diag_update` is None, implying D=I
    base_matrix_4 = np.identity(2, dtype=np.float32)
    base_op_4 = tf.linalg.LinearOperatorFullMatrix(
        matrix=base_matrix_4,
        is_non_singular=True, is_self_adjoint=True,
        is_positive_definite=True, is_square=True)
    base_op_4.size = base_matrix_4.size
    input_dict_4 = {
        'base_operator': base_op_4,
        'u': np.array([[1.], [2.]], dtype=np.float32),
        'diag_update': None,
        'v': np.array([[2.], [1.]], dtype=np.float32),
        'is_diag_update_positive': None,
        'is_non_singular': None,
        'is_self_adjoint': None,
        'is_positive_definite': None,
        'is_square': True,
        'name': 'diag_update_is_none'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Batched operator with LinearOperatorDiag base (replaces faulty rectangular case)
    base_diag_5 = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    base_op_5 = tf.linalg.LinearOperatorDiag(
        diag=base_diag_5,
        is_non_singular=True, is_self_adjoint=True,
        is_positive_definite=True, is_square=True)
    base_op_5.size = base_diag_5.size
    input_dict_5 = {
        'base_operator': base_op_5,
        'u': np.random.rand(2, 2, 1).astype(np.float32),
        'diag_update': np.random.rand(2, 1).astype(np.float32),
        'v': None,
        'is_diag_update_positive': None,
        'is_non_singular': None,
        'is_self_adjoint': True,
        'is_positive_definite': None,
        'is_square': True,
        'name': 'batched_diag_base'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Batched square operator (B=2, M=N=3, K=1)
    base_matrix_6 = np.array([np.identity(3), np.identity(3) * 2], dtype=np.float32)
    base_op_6 = tf.linalg.LinearOperatorFullMatrix(
        matrix=base_matrix_6,
        is_non_singular=True, is_self_adjoint=True,
        is_positive_definite=True, is_square=True)
    base_op_6.size = base_matrix_6.size
    input_dict_6 = {
        'base_operator': base_op_6,
        'u': np.random.rand(2, 3, 1).astype(np.float32),
        'diag_update': np.random.rand(2, 1).astype(np.float32),
        'v': np.random.rand(2, 3, 1).astype(np.float32),
        'is_diag_update_positive': None,
        'is_non_singular': None,
        'is_self_adjoint': None,
        'is_positive_definite': None,
        'is_square': True,
        'name': 'batched_square'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Negative definite case
    base_matrix_7 = np.array([[-1, 0], [0, -2]], dtype=np.float32)
    base_op_7 = tf.linalg.LinearOperatorFullMatrix(
        matrix=base_matrix_7,
        is_non_singular=True, is_self_adjoint=True,
        is_positive_definite=False, is_square=True)
    base_op_7.size = base_matrix_7.size
    input_dict_7 = {
        'base_operator': base_op_7,
        'u': np.array([[1.], [1.]], dtype=np.float32),
        'diag_update': np.array([-3.], dtype=np.float32),
        'v': None,
        'is_diag_update_positive': False,
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'negative_definite'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Float64 data type
    base_matrix_8 = np.identity(3, dtype=np.float64)
    base_op_8 = tf.linalg.LinearOperatorFullMatrix(
        matrix=base_matrix_8,
        is_non_singular=True, is_self_adjoint=True,
        is_positive_definite=True, is_square=True)
    base_op_8.size = base_matrix_8.size
    input_dict_8 = {
        'base_operator': base_op_8,
        'u': np.array([[1.1, 2.2], [-1.1, 3.3], [0., 1.]], dtype=np.float64),
        'diag_update': np.array([11.1, 12.2], dtype=np.float64),
        'v': None,
        'is_diag_update_positive': True,
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'float64_type'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Complex data type (general non-hermitian)
    base_matrix_9 = np.array([[1, 2j], [3j, 4]], dtype=np.complex128)
    base_op_9 = tf.linalg.LinearOperatorFullMatrix(
        matrix=base_matrix_9,
        is_non_singular=True, is_self_adjoint=False,
        is_positive_definite=False, is_square=True)
    base_op_9.size = base_matrix_9.size
    input_dict_9 = {
        'base_operator': base_op_9,
        'u': np.array([[1+1j], [2-2j]], dtype=np.complex128),
        'diag_update': np.array([1+3j], dtype=np.complex128),
        'v': np.array([[5], [6j]], dtype=np.complex128),
        'is_diag_update_positive': None,
        'is_non_singular': None,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'complex_general'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Complex data type (Hermitian)
    base_matrix_10 = np.array([[2, 1j], [-1j, 2]], dtype=np.complex64)
    base_op_10 = tf.linalg.LinearOperatorFullMatrix(
        matrix=base_matrix_10,
        is_non_singular=True, is_self_adjoint=True,
        is_positive_definite=True, is_square=True)
    base_op_10.size = base_matrix_10.size
    input_dict_10 = {
        'base_operator': base_op_10,
        'u': np.array([[1+1j], [2-2j]], dtype=np.complex64),
        'diag_update': np.array([3.], dtype=np.float32),
        'v': None,
        'is_diag_update_positive': True,
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'complex_hermitian_pd'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.linalg.LinearOperatorLowRankUpdate"] = tf_linalg_linearoperatorlowrankupdate_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.linalg.LinearOperatorLowRankUpdate' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorLowRankUpdate'.")

check_valid('tf.linalg.LinearOperatorLowRankUpdate', generated_inputs['tf.linalg.LinearOperatorLowRankUpdate'], lib="tf", suffix=0)
