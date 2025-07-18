
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
import functools

@functools.total_ordering
class ComparableLinearOperatorFullMatrix(tf.linalg.LinearOperatorFullMatrix):
    def __eq__(self, other):
        if not isinstance(other, ComparableLinearOperatorFullMatrix):
            return NotImplemented
        return id(self) == id(other)

    def __lt__(self, other):
        if not isinstance(other, ComparableLinearOperatorFullMatrix):
            return NotImplemented
        return id(self) < id(other)

def tf_linalg_linearoperatorblockdiag_inputs():
    """
    Returns a list of valid inputs for tf.linalg.LinearOperatorBlockDiag.
    """
    list_of_inputs = []

    # Input 1: Basic case with two 2x2 float32 operators
    op1 = ComparableLinearOperatorFullMatrix(np.array([[1., 2.], [3., 4.]], dtype=np.float32))
    op2 = ComparableLinearOperatorFullMatrix(np.array([[5., 6.], [7., 8.]], dtype=np.float32))
    input_dict_1 = {
        'operators': [op1, op2],
        'is_non_singular': None,
        'is_self_adjoint': None,
        'is_positive_definite': None,
        'is_square': True,
        'name': 'basic_2x2_float32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Non-square operators resulting in a non-square block operator
    op1 = ComparableLinearOperatorFullMatrix(np.array([[1.], [2.], [3.]], dtype=np.float32))
    op2 = ComparableLinearOperatorFullMatrix(np.array([[4., 5.]], dtype=np.float32))
    input_dict_2 = {
        'operators': [op1, op2],
        'is_non_singular': None,
        'is_self_adjoint': None,
        'is_positive_definite': None,
        'is_square': False,
        'name': 'non_square'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Batched operators with broadcasting batch shapes
    mat1 = np.random.rand(2, 1, 3, 3).astype(np.float32)
    mat2 = np.random.rand(1, 4, 2, 2).astype(np.float32)
    op1 = ComparableLinearOperatorFullMatrix(mat1)
    op2 = ComparableLinearOperatorFullMatrix(mat2)
    input_dict_3 = {
        'operators': [op1, op2],
        'is_non_singular': None,
        'is_self_adjoint': None,
        'is_positive_definite': None,
        'is_square': True,
        'name': 'batched_ops_broadcast'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: A single operator in the list, with positive definite hints
    op1 = ComparableLinearOperatorFullMatrix(np.eye(4, dtype=np.float64))
    input_dict_4 = {
        'operators': [op1],
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'single_positive_definite_op'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: More than two operators with mixed shapes
    op1 = ComparableLinearOperatorFullMatrix(np.array([[1., 0.], [0., 1.]], dtype=np.float32))
    op2 = ComparableLinearOperatorFullMatrix(np.array([[2.]], dtype=np.float32))
    op3 = ComparableLinearOperatorFullMatrix(np.array([[3., 4., 5.]], dtype=np.float32))
    input_dict_5 = {
        'operators': [op1, op2, op3],
        'is_non_singular': None,
        'is_self_adjoint': None,
        'is_positive_definite': None,
        'is_square': False,
        'name': 'three_ops_mixed_shape'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: float64 dtype with a non-singular hint
    op1 = ComparableLinearOperatorFullMatrix(np.array([[1., 2.], [-2., 1.]], dtype=np.float64))
    op2 = ComparableLinearOperatorFullMatrix(np.array([[10., 0.], [0., 10.]], dtype=np.float64))
    input_dict_6 = {
        'operators': [op1, op2],
        'is_non_singular': True,
        'is_self_adjoint': None,
        'is_positive_definite': None,
        'is_square': True,
        'name': 'float64_non_singular'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Self-adjoint hint for symmetric operators
    mat1 = np.array([[1., 2.], [2., 1.]], dtype=np.float32)
    mat2 = np.array([[5., -1.], [-1., 5.]], dtype=np.float32)
    op1 = ComparableLinearOperatorFullMatrix(mat1)
    op2 = ComparableLinearOperatorFullMatrix(mat2)
    input_dict_7 = {
        'operators': [op1, op2],
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': None,
        'is_square': True,
        'name': 'self_adjoint_hint'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Positive-definite hint for positive-definite operators
    mat1 = np.array([[2., 1.], [1., 2.]], dtype=np.float32)
    mat2 = np.eye(3, dtype=np.float32) * 4
    op1 = ComparableLinearOperatorFullMatrix(mat1)
    op2 = ComparableLinearOperatorFullMatrix(mat2)
    input_dict_8 = {
        'operators': [op1, op2],
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'positive_definite_hint'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: All relevant boolean hints set to False
    op1 = ComparableLinearOperatorFullMatrix(np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32))
    op2 = ComparableLinearOperatorFullMatrix(np.array([[1., 1.], [1., 1.]], dtype=np.float32))
    input_dict_9 = {
        'operators': [op1, op2],
        'is_non_singular': False,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': False,
        'name': 'all_flags_false'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Batched operators with identical batch shapes
    mat1 = np.random.rand(5, 2, 2).astype(np.float32)
    mat2 = np.random.rand(5, 3, 3).astype(np.float32)
    op1 = ComparableLinearOperatorFullMatrix(mat1)
    op2 = ComparableLinearOperatorFullMatrix(mat2)
    input_dict_10 = {
        'operators': [op1, op2],
        'is_non_singular': None,
        'is_self_adjoint': None,
        'is_positive_definite': None,
        'is_square': True,
        'name': 'batched_same_batch_shape'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

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
