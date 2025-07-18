
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

# Helper classes to add a .size property to LinearOperator, working around a
# test harness limitation that caused the first error. This allows passing the
# correct object type (LinearOperator) to the API, which fixes the second error.
class LinearOperatorDiagWithSize(tf.linalg.LinearOperatorDiag):
    @property
    def size(self):
        # The test harness expects a .size attribute.
        return np.prod(self.shape.as_list())

class LinearOperatorFullMatrixWithSize(tf.linalg.LinearOperatorFullMatrix):
    @property
    def size(self):
        # The test harness expects a .size attribute.
        return np.prod(self.shape.as_list())


def get_tf_linalg_linearoperatorlowrankupdate_inputs():
    """
    Generate a list of valid inputs for tf.linalg.LinearOperatorLowRankUpdate.
    """
    list_of_inputs = []

    # Input 1: Basic case from docs, square matrix
    base_op_1 = LinearOperatorDiagWithSize(
        diag=np.array([1., 2., 3.], dtype=np.float32), is_non_singular=True, is_self_adjoint=True,
        is_positive_definite=True)
    input_dict_1 = {
        'base_operator': base_op_1,
        'u': np.array([[1., 2.], [-1., 3.], [0., 0.]], dtype=np.float32),
        'diag_update': np.array([11., 12.], dtype=np.float32),
        'v': np.array([[1., 2.], [-1., 3.], [10., 10.]], dtype=np.float32),
        'is_diag_update_positive': None,
        'is_non_singular': None,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'basic_square_update'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Self-adjoint update (v=None implies v=u)
    base_op_2 = LinearOperatorFullMatrixWithSize(
        np.array([[2, 1, 0], [1, 2, 1], [0, 1, 2]], dtype=np.float32),
        is_self_adjoint=True, is_positive_definite=True)
    input_dict_2 = {
        'base_operator': base_op_2,
        'u': np.array([[1.], [2.], [3.]], dtype=np.float32),
        'diag_update': np.array([5.], dtype=np.float32),
        'v': None,
        'is_diag_update_positive': True,
        'is_non_singular': None,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'self_adjoint_update'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Positive-definite update
    base_op_3 = LinearOperatorDiagWithSize(
        np.array([1., 2., 3.], dtype=np.float32), is_positive_definite=True, is_self_adjoint=True)
    input_dict_3 = {
        'base_operator': base_op_3,
        'u': np.array([[1.], [0.], [1.]], dtype=np.float32),
        'diag_update': np.array([4.], dtype=np.float32),
        'v': None,
        'is_diag_update_positive': True,
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'positive_definite_update'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Square matrix (modified from non-square to avoid broadcast error)
    base_op_4 = LinearOperatorFullMatrixWithSize(np.arange(16, dtype=np.float32).reshape(4, 4))
    input_dict_4 = {
        'base_operator': base_op_4,
        'u': np.random.rand(4, 2).astype(np.float32),
        'diag_update': np.array([1., 2.], dtype=np.float32),
        'v': np.random.rand(4, 2).astype(np.float32),
        'is_diag_update_positive': None,
        'is_non_singular': None,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'square_update_from_non_square'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Batched operator
    base_op_5 = LinearOperatorDiagWithSize(
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32), is_self_adjoint=True)
    input_dict_5 = {
        'base_operator': base_op_5,
        'u': np.random.rand(2, 3, 1).astype(np.float32),
        'diag_update': np.array([[10], [20]], dtype=np.float32),
        'v': None,
        'is_diag_update_positive': True,
        'is_non_singular': None,
        'is_self_adjoint': True,
        'is_positive_definite': None,
        'is_square': True,
        'name': 'batched_update'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Minimal flags (all boolean flags are None)
    base_op_6 = LinearOperatorFullMatrixWithSize(np.array([[1., 0.], [0., 1.]], dtype=np.float32), is_self_adjoint=True, is_positive_definite=True)
    input_dict_6 = {
        'base_operator': base_op_6,
        'u': np.array([[1.], [1.]], dtype=np.float32),
        'diag_update': np.array([1.], dtype=np.float32),
        'v': None,
        'is_diag_update_positive': None,
        'is_non_singular': None,
        'is_self_adjoint': None,
        'is_positive_definite': None,
        'is_square': None,
        'name': 'minimal_flags'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Negative values in tensors
    base_op_7 = LinearOperatorDiagWithSize(np.array([-1., -2., -3.], dtype=np.float32), is_self_adjoint=True)
    input_dict_7 = {
        'base_operator': base_op_7,
        'u': np.array([[-1., 2.], [1., -3.], [0., 0.]], dtype=np.float32),
        'diag_update': np.array([-11., 12.], dtype=np.float32),
        'v': np.array([[1., -2.], [-1., 3.], [-10., 10.]], dtype=np.float32),
        'is_diag_update_positive': False,
        'is_non_singular': None,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'negative_values_update'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: float64 dtype
    base_op_8 = LinearOperatorDiagWithSize(np.array([1e10, 2e10], dtype=np.float64), is_self_adjoint=True)
    input_dict_8 = {
        'base_operator': base_op_8,
        'u': np.array([[1e-5], [2e-5]], dtype=np.float64),
        'diag_update': np.array([-5.], dtype=np.float64),
        'v': None,
        'is_diag_update_positive': False,
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'float64_update'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Complex numbers
    base_mat_9 = np.array([[1+1j, 2-3j], [2+3j, 5+0j]], dtype=np.complex64) # Hermitian matrix
    base_op_9 = LinearOperatorFullMatrixWithSize(base_mat_9, is_self_adjoint=True)
    input_dict_9 = {
        'base_operator': base_op_9,
        'u': np.array([[1+2j], [3-1j]], dtype=np.complex64),
        'diag_update': np.array([2.+0.j], dtype=np.complex64),
        'v': None,
        'is_diag_update_positive': None,
        'is_non_singular': None,
        'is_self_adjoint': True,
        'is_positive_definite': None,
        'is_square': True,
        'name': 'complex_self_adjoint_update'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Batched square operator (modified from non-square)
    base_mats_10 = np.arange(32, dtype=np.float32).reshape(2, 4, 4)
    base_op_10 = LinearOperatorFullMatrixWithSize(base_mats_10)
    input_dict_10 = {
        'base_operator': base_op_10,
        'u': np.random.rand(2, 4, 1).astype(np.float32),
        'diag_update': np.array([[1.], [-1.]], dtype=np.float32),
        'v': np.random.rand(2, 4, 1).astype(np.float32),
        'is_diag_update_positive': False,
        'is_non_singular': None,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'batched_square_from_non_square'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.LinearOperatorLowRankUpdate"] = get_tf_linalg_linearoperatorlowrankupdate_inputs()

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
