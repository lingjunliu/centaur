
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_linearoperatorlowrankupdate_inputs():
    """
    Generates a list of valid inputs for the tf.linalg.LinearOperatorLowRankUpdate function.
    """
    list_of_inputs = []

    # Case 1: Base operator is self-adjoint and positive-definite.
    base_op_1 = np.array([[2., 0.], [0., 3.]], dtype=np.float32)
    u_1 = np.array([[1.], [1.]], dtype=np.float32)
    input_dict_1 = {
        'base_operator': base_op_1,
        'u': u_1,
        'diag_update': np.array([1.], dtype=np.float32),
        'v': u_1,
        'is_diag_update_positive': True,
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'simple_pd_sa'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Case 2: Base operator is self-adjoint but not positive-definite.
    base_op_2 = np.array([[2., 1.], [1., -1.]], dtype=np.float32)
    u_2 = np.array([[1., 0.], [0., 1.]], dtype=np.float32)
    input_dict_2 = {
        'base_operator': base_op_2,
        'u': u_2,
        'diag_update': np.array([1., 1.], dtype=np.float32),
        'v': u_2,
        'is_diag_update_positive': True,
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'simple_sa_not_pd'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Case 3: Base operator is not self-adjoint.
    base_op_3 = np.array([[1., 2.], [3., 4.]], dtype=np.float64)
    input_dict_3 = {
        'base_operator': base_op_3,
        'u': np.array([[1.], [2.]], dtype=np.float64),
        'diag_update': np.array([-5.], dtype=np.float64),
        'v': np.array([[3.], [4.]], dtype=np.float64),
        'is_diag_update_positive': False,
        'is_non_singular': True,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'simple_non_sa'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Case 4: Batched operation.
    base_op_4 = np.array([[[2., 0.], [0., 3.]], [[4., 0.], [0., 5.]]], dtype=np.float32)
    u_4 = np.random.randn(2, 2, 1).astype(np.float32)
    input_dict_4 = {
        'base_operator': base_op_4,
        'u': u_4,
        'diag_update': np.array([[1.], [1.]], dtype=np.float32),
        'v': u_4,
        'is_diag_update_positive': True,
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'batched_pd_sa'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Case 5: Non-square operator.
    base_op_5 = np.random.randn(3, 4).astype(np.float32)
    input_dict_5 = {
        'base_operator': base_op_5,
        'u': np.random.randn(3, 2).astype(np.float32),
        'diag_update': np.array([1., 1.], dtype=np.float32),
        'v': np.random.randn(4, 2).astype(np.float32),
        'is_diag_update_positive': True,
        'is_non_singular': False,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': False,
        'name': 'non_square'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))
    
    # Case 6: Singular base operator.
    base_op_6 = np.array([[1., 1.], [1., 1.]], dtype=np.float32)
    input_dict_6 = {
        'base_operator': base_op_6,
        'u': np.array([[1.], [0.]], dtype=np.float32),
        'diag_update': np.array([1.], dtype=np.float32),
        'v': np.array([[0.], [1.]], dtype=np.float32),
        'is_diag_update_positive': True,
        'is_non_singular': True,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'singular_base'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Case 7: Complex valued, non-hermitian
    base_op_7 = (np.random.randn(2, 2) + 1j * np.random.randn(2,2)).astype(np.complex64)
    u_7 = (np.random.randn(2, 1) + 1j * np.random.randn(2,1)).astype(np.complex64)
    v_7 = (np.random.randn(2, 1) + 1j * np.random.randn(2,1)).astype(np.complex64)
    input_dict_7 = {
        'base_operator': base_op_7,
        'u': u_7,
        'diag_update': np.array([1.+1.j], dtype=np.complex64),
        'v': v_7,
        'is_diag_update_positive': False,
        'is_non_singular': True,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'complex_non_hermitian'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

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
