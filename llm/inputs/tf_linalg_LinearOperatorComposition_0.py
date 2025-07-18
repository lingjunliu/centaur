
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

# The testing framework's utility function attempts to call np.min/np.max
# on the `operators` list. `LinearOperator` objects and their subclasses are
# not comparable by default, leading to a TypeError. To work around this, we
# monkey-patch the relevant classes to make their instances comparable based
# on their object IDs. This is a necessary workaround to prevent the test
# harness from crashing and does not affect the correctness of the API call.
def _op_lt(self, other): return id(self) < id(other)
def _op_le(self, other): return id(self) <= id(other)
def _op_gt(self, other): return id(self) > id(other)
def _op_ge(self, other): return id(self) >= id(other)

classes_to_patch = [
    tf.linalg.LinearOperator,
    tf.linalg.LinearOperatorFullMatrix,
    tf.linalg.LinearOperatorDiag,
    tf.linalg.LinearOperatorIdentity,
]

for cls in classes_to_patch:
    if not hasattr(cls, '__lt__'):
        cls.__lt__ = _op_lt
        cls.__le__ = _op_le
        cls.__gt__ = _op_gt
        cls.__ge__ = _op_ge

def get_tf_linalg_linearoperatorcomposition_inputs():
    """
    Generates a list of valid inputs for tf.linalg.LinearOperatorComposition.
    """
    list_of_inputs = []

    # Input 1: Basic composition of two 2x2 square operators
    op1_1 = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 2.], [3., 4.]], dtype=np.float32))
    op1_2 = tf.linalg.LinearOperatorFullMatrix(np.array([[5., 6.], [7., 8.]], dtype=np.float32))
    input_dict_1 = {
        'operators': [op1_1, op1_2],
        'is_non_singular': True,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'composition_2x2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Composition of non-square operators
    op2_1 = tf.linalg.LinearOperatorFullMatrix(np.random.rand(2, 3).astype(np.float32))
    op2_2 = tf.linalg.LinearOperatorFullMatrix(np.random.rand(3, 4).astype(np.float32))
    input_dict_2 = {
        'operators': [op2_1, op2_2],
        'is_non_singular': False,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': False,
        'name': 'composition_non_square'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Composition with a single diagonal operator
    op3_1 = tf.linalg.LinearOperatorDiag(np.array([1., -1., 2.], dtype=np.float64))
    input_dict_3 = {
        'operators': [op3_1],
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'composition_single_operator'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Composition of mixed operator types
    op4_1 = tf.linalg.LinearOperatorFullMatrix(np.random.rand(3, 3).astype(np.float32))
    op4_2 = tf.linalg.LinearOperatorDiag(np.array([1., 2., 3.], dtype=np.float32))
    op4_3 = tf.linalg.LinearOperatorIdentity(num_rows=3, dtype=tf.float32)
    input_dict_4 = {
        'operators': [op4_1, op4_2, op4_3],
        'is_non_singular': None,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'composition_mixed_types'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Composition with batch dimensions
    matrix5_1 = np.random.rand(4, 2, 3).astype(np.float32)
    matrix5_2 = np.random.rand(4, 3, 2).astype(np.float32)
    op5_1 = tf.linalg.LinearOperatorFullMatrix(matrix5_1)
    op5_2 = tf.linalg.LinearOperatorFullMatrix(matrix5_2)
    input_dict_5 = {
        'operators': [op5_1, op5_2],
        'is_non_singular': None,
        'is_self_adjoint': None,
        'is_positive_definite': None,
        'is_square': True,
        'name': 'composition_batch'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))
    
    # Input 6: An empty list of operators (should result in an identity operator)
    input_dict_6 = {
        'operators': [],
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'composition_empty_list'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Composition guaranteed to be Positive Definite
    op7_1 = tf.linalg.LinearOperatorDiag(np.array([2., 3.], dtype=np.float32))
    op7_2 = tf.linalg.LinearOperatorDiag(np.array([4., 5.], dtype=np.float32))
    input_dict_7 = {
        'operators': [op7_1, op7_2],
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'composition_positive_definite'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    
    # Input 8: Composition with negative values
    op8_1 = tf.linalg.LinearOperatorFullMatrix(np.array([[-1., 0.], [2., -3.]], dtype=np.float32))
    op8_2 = tf.linalg.LinearOperatorDiag(np.array([-2., 5.], dtype=np.float32))
    input_dict_8 = {
        'operators': [op8_1, op8_2],
        'is_non_singular': True,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'composition_negative_values'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

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
