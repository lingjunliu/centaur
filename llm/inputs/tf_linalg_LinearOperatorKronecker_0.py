
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_LinearOperatorKronecker_inputs():
    """
    Generates a list of valid inputs for tf.linalg.LinearOperatorKronecker.
    """
    # Helper class to work around a validation issue where lists of non-comparable
    # objects cause a TypeError. This makes the LinearOperator objects comparable.
    class _ComparableLinearOperatorFullMatrix(tf.linalg.LinearOperatorFullMatrix):
        def __lt__(self, other):
            return True
        def __le__(self, other):
            return True
        def __gt__(self, other):
            return True
        def __ge__(self, other):
            return True

    list_of_inputs = []

    # Input 1: Basic case with two 2x2 square operators
    op1_mat_1 = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    op2_mat_1 = np.array([[5., 6.], [7., 8.]], dtype=np.float32)
    input_dict_1 = {
        'operators': [
            _ComparableLinearOperatorFullMatrix(op1_mat_1),
            _ComparableLinearOperatorFullMatrix(op2_mat_1)
        ],
        'is_non_singular': True,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'basic_kronecker_2x2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Three operators, all positive-definite and self-adjoint
    op1_mat_2 = np.array([[1., 0.], [0., 1.]], dtype=np.float32)
    op2_mat_2 = np.array([[2.]], dtype=np.float32)
    op3_mat_2 = np.array([[3., 1.], [1., 3.]], dtype=np.float32)
    input_dict_2 = {
        'operators': [
            _ComparableLinearOperatorFullMatrix(op1_mat_2),
            _ComparableLinearOperatorFullMatrix(op2_mat_2),
            _ComparableLinearOperatorFullMatrix(op3_mat_2)
        ],
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'three_operators_pd_sa'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Non-square operators
    op1_mat_3 = np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32) # 2x3
    op2_mat_3 = np.array([[1., 0.], [0., 1.]], dtype=np.float32) # 2x2
    input_dict_3 = {
        'operators': [
            _ComparableLinearOperatorFullMatrix(op1_mat_3),
            _ComparableLinearOperatorFullMatrix(op2_mat_3)
        ],
        'is_non_singular': False,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': False,
        'name': 'non_square_kronecker'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Batch dimension
    op1_mat_4 = np.random.rand(3, 2, 2).astype(np.float32)
    op2_mat_4 = np.random.rand(3, 2, 3).astype(np.float32)
    input_dict_4 = {
        'operators': [
            _ComparableLinearOperatorFullMatrix(op1_mat_4),
            _ComparableLinearOperatorFullMatrix(op2_mat_4)
        ],
        'is_non_singular': None,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': False,
        'name': 'batched_kronecker'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Broadcastable batch dimensions
    op1_mat_5 = np.eye(2, dtype=np.float32)[np.newaxis, np.newaxis, :, :]
    op2_mat_5 = np.random.rand(4, 5, 3, 3).astype(np.float32)
    input_dict_5 = {
        'operators': [
            _ComparableLinearOperatorFullMatrix(op1_mat_5),
            _ComparableLinearOperatorFullMatrix(op2_mat_5)
        ],
        'is_non_singular': None,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'broadcast_batch_kronecker'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: float64 dtype and negative values
    op1_mat_6 = np.array([[-1., -2.], [-3., -4.]], dtype=np.float64)
    op2_mat_6 = np.array([[1., 0., 0.], [0., 1., 0.]], dtype=np.float64)
    input_dict_6 = {
        'operators': [
            _ComparableLinearOperatorFullMatrix(op1_mat_6),
            _ComparableLinearOperatorFullMatrix(op2_mat_6)
        ],
        'is_non_singular': False,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': False,
        'name': 'float64_kronecker_negative'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Singular operator in the product
    op1_mat_7 = np.ones((2, 2), dtype=np.float32)
    op2_mat_7 = np.eye(3, dtype=np.float32)
    input_dict_7 = {
        'operators': [
            _ComparableLinearOperatorFullMatrix(op1_mat_7),
            _ComparableLinearOperatorFullMatrix(op2_mat_7)
        ],
        'is_non_singular': False,
        'is_self_adjoint': True,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'singular_kronecker'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Batched identity operators
    op1_mat_8 = np.eye(2, dtype=np.float32)[np.newaxis, :, :]
    op2_mat_8 = np.eye(3, dtype=np.float32)[np.newaxis, :, :]
    input_dict_8 = {
        'operators': [
            _ComparableLinearOperatorFullMatrix(op1_mat_8),
            _ComparableLinearOperatorFullMatrix(op2_mat_8)
        ],
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'batched_identity'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: All flags explicitly None
    op1_mat_9 = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    op2_mat_9 = np.array([[5., 6.], [7., 8.]], dtype=np.float32)
    input_dict_9 = {
        'operators': [
            _ComparableLinearOperatorFullMatrix(op1_mat_9),
            _ComparableLinearOperatorFullMatrix(op2_mat_9)
        ],
        'is_non_singular': None,
        'is_self_adjoint': None,
        'is_positive_definite': None,
        'is_square': None,
        'name': 'all_flags_none'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Larger number of operators
    op1_mat_10 = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    op2_mat_10 = np.array([[1., 0.], [2., 1.]], dtype=np.float32)
    op3_mat_10 = np.eye(2, dtype=np.float32)
    op4_mat_10 = np.array([[2., 0.], [0., 0.5]], dtype=np.float32)
    input_dict_10 = {
        'operators': [
            _ComparableLinearOperatorFullMatrix(op1_mat_10),
            _ComparableLinearOperatorFullMatrix(op2_mat_10),
            _ComparableLinearOperatorFullMatrix(op3_mat_10),
            _ComparableLinearOperatorFullMatrix(op4_mat_10)
        ],
        'is_non_singular': True,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'four_operators'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.linalg.LinearOperatorKronecker"] = tf_linalg_LinearOperatorKronecker_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.linalg.LinearOperatorKronecker' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorKronecker'.")

check_valid('tf.linalg.LinearOperatorKronecker', generated_inputs['tf.linalg.LinearOperatorKronecker'], lib="tf", suffix=0)
