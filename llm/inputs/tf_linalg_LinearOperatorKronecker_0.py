
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

# Helper classes to make LinearOperator instances comparable for the validation script.
class ComparableMixin:
    """A mixin to make objects fully comparable based on their id."""
    def __lt__(self, other):
        return id(self) < id(other)
    def __le__(self, other):
        return id(self) <= id(other)
    def __gt__(self, other):
        return id(self) > id(other)
    def __ge__(self, other):
        return id(self) >= id(other)

class ComparableLinearOperatorFullMatrix(ComparableMixin, tf.linalg.LinearOperatorFullMatrix):
    pass

class ComparableLinearOperatorIdentity(ComparableMixin, tf.linalg.LinearOperatorIdentity):
    pass

class ComparableLinearOperatorDiag(ComparableMixin, tf.linalg.LinearOperatorDiag):
    pass

class ComparableLinearOperatorLowerTriangular(ComparableMixin, tf.linalg.LinearOperatorLowerTriangular):
    pass


def get_tf_linalg_linearoperatorkronecker_inputs():
    """
    Generates a list of valid inputs for tf.linalg.LinearOperatorKronecker.
    """
    list_of_inputs = []

    # Input 1: Basic case with two 2x2 operators
    op1_matrix = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    op2_matrix = np.array([[5., 6.], [7., 8.]], dtype=np.float32)
    operator_1 = ComparableLinearOperatorFullMatrix(op1_matrix)
    operator_2 = ComparableLinearOperatorFullMatrix(op2_matrix)
    input_dict_1 = {
        'operators': [operator_1, operator_2],
        'is_non_singular': False,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'basic_2x2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Three operators resulting in a non-square matrix
    op1 = ComparableLinearOperatorFullMatrix(np.random.rand(2, 3).astype(np.float32))
    op2 = ComparableLinearOperatorFullMatrix(np.random.rand(3, 2).astype(np.float32))
    op3 = ComparableLinearOperatorFullMatrix(np.random.rand(4, 5).astype(np.float32))
    input_dict_2 = {
        'operators': [op1, op2, op3],
        'is_non_singular': False,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': False,
        'name': 'three_non_square'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Operators with a batch dimension
    matrix1 = np.random.rand(2, 3, 3).astype(np.float32)
    matrix2 = np.random.rand(2, 2, 2).astype(np.float32)
    op1 = ComparableLinearOperatorFullMatrix(matrix1)
    op2 = ComparableLinearOperatorFullMatrix(matrix2)
    input_dict_3 = {
        'operators': [op1, op2],
        'is_non_singular': False,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'with_batch_dim'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Broadcasting batch dimensions
    matrix1_bcast = np.random.rand(3, 1, 2, 2).astype(np.float32)
    matrix2_bcast = np.random.rand(1, 4, 3, 3).astype(np.float32)
    op1_bcast = ComparableLinearOperatorFullMatrix(matrix1_bcast)
    op2_bcast = ComparableLinearOperatorFullMatrix(matrix2_bcast)
    input_dict_4 = {
        'operators': [op1_bcast, op2_bcast],
        'is_non_singular': False,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'broadcast_batch'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Mix of different LinearOperator types
    op_id = ComparableLinearOperatorIdentity(2, dtype=np.float32)
    op_diag = ComparableLinearOperatorDiag(np.array([1., -1.], dtype=np.float32))
    op_full = ComparableLinearOperatorFullMatrix(np.array([[1., 0.], [2., 1.]], dtype=np.float32))
    input_dict_5 = {
        'operators': [op_id, op_diag, op_full],
        'is_non_singular': True,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'mixed_operator_types'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Using float64 dtype
    op1_f64 = ComparableLinearOperatorFullMatrix(np.array([[1., 0.], [0., 1.]], dtype=np.float64))
    op2_f64 = ComparableLinearOperatorFullMatrix(np.array([[2., 3.], [3., 2.]], dtype=np.float64))
    input_dict_6 = {
        'operators': [op1_f64, op2_f64],
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'float64_dtype'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Input 7: Using complex64 dtype
    op1_c64 = ComparableLinearOperatorFullMatrix(np.array([[1+1j, 2-1j], [3+0j, 4-2j]], dtype=np.complex64))
    op2_c64 = ComparableLinearOperatorIdentity(3, dtype=np.complex64)
    input_dict_7 = {
        'operators': [op1_c64, op2_c64],
        'is_non_singular': True,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'complex64_dtype'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: True property hints (self-adjoint, positive-definite)
    spd_mat1 = np.array([[4., 1.], [1., 3.]], dtype=np.float32)
    spd_mat2 = np.array([[2., 0.], [0., 5.]], dtype=np.float32)
    op1_spd = ComparableLinearOperatorFullMatrix(spd_mat1, is_self_adjoint=True, is_positive_definite=True)
    op2_spd = ComparableLinearOperatorFullMatrix(spd_mat2, is_self_adjoint=True, is_positive_definite=True)
    input_dict_8 = {
        'operators': [op1_spd, op2_spd],
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'spd_hints'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Singular operator case
    singular_mat = np.array([[1., 1.], [1., 1.]], dtype=np.float32)
    op1_singular = ComparableLinearOperatorFullMatrix(singular_mat)
    op2_singular = ComparableLinearOperatorIdentity(2, dtype=np.float32)
    input_dict_9 = {
        'operators': [op1_singular, op2_singular],
        'is_non_singular': False,
        'is_self_adjoint': True,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'singular_case'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Larger number of operators in the list
    ops = [ComparableLinearOperatorFullMatrix(np.random.rand(2, 2).astype(np.float32)) for _ in range(4)]
    input_dict_10 = {
        'operators': ops,
        'is_non_singular': False,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'four_operators'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Using negative values
    op1_neg = ComparableLinearOperatorFullMatrix(np.array([[-1., -2.], [-3., -4.]], dtype=np.float32))
    op2_neg = ComparableLinearOperatorDiag(np.array([1., -1.], dtype=np.float32))
    input_dict_11 = {
        'operators': [op1_neg, op2_neg],
        'is_non_singular': True,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'negative_values'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    # Input 12: Using LinearOperatorLowerTriangular
    tril1 = ComparableLinearOperatorLowerTriangular(np.array([[1., 0.], [2., 3.]], dtype=np.float32))
    tril2 = ComparableLinearOperatorLowerTriangular(np.array([[4., 0.], [5., 6.]], dtype=np.float32))
    input_dict_12 = {
        'operators': [tril1, tril2],
        'is_non_singular': True,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'lower_triangular'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_12))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.LinearOperatorKronecker"] = get_tf_linalg_linearoperatorkronecker_inputs()

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
