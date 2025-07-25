
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

# Helper class to make LinearOperator objects comparable and deep-copyable for the validation framework
class ComparableLinearOperator:
    def __init__(self, op, identifier):
        self._op = op
        self._id = identifier

    def __getattr__(self, name):
        # Delegate attribute access to the wrapped LinearOperator
        return getattr(self._op, name)

    # Implement comparison operators for the validation script
    def __lt__(self, other): return self._id < other._id
    def __le__(self, other): return self._id <= other._id
    def __gt__(self, other): return self._id > other._id
    def __ge__(self, other): return self._id >= other._id
    def __eq__(self, other): return self._id == other._id
    def __ne__(self, other): return self._id != other._id
    
    # Implement deepcopy to avoid errors with TensorFlow objects
    def __deepcopy__(self, memo):
        # Create a new instance of the wrapper but pass the same tf.Operator object reference.
        if id(self) in memo:
            return memo[id(self)]
        
        new_op_wrapper = ComparableLinearOperator(self._op, self._id)
        memo[id(self)] = new_op_wrapper
        return new_op_wrapper

def get_linear_operator_kronecker_inputs():
    list_of_inputs = []

    # Input 1: Basic case with two 2x2 operators
    op1_1 = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 2.], [3., 4.]], dtype=np.float32))
    op1_2 = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 0.], [2., 1.]], dtype=np.float32))
    input_1 = {
        'operators': [ComparableLinearOperator(op1_1, 0), ComparableLinearOperator(op1_2, 1)],
        'is_non_singular': None, 'is_self_adjoint': None, 'is_positive_definite': None, 'is_square': None, 'name': 'basic_wrapped'
    }
    list_of_inputs.append(copy.deepcopy(input_1))

    # Input 2: Three operators, demonstrating inference
    op2_1 = tf.linalg.LinearOperatorIdentity(num_rows=2, dtype=tf.float32)
    op2_2 = tf.linalg.LinearOperatorFullMatrix(np.array([[2., 0.], [0., 2.]], dtype=np.float32))
    op2_3 = tf.linalg.LinearOperatorFullMatrix(np.array([[3., 1.], [1., 3.]], dtype=np.float32))
    input_2 = {
        'operators': [ComparableLinearOperator(op2_1, 0), ComparableLinearOperator(op2_2, 1), ComparableLinearOperator(op2_3, 2)],
        'is_non_singular': None, 'is_self_adjoint': None, 'is_positive_definite': None, 'is_square': None, 'name': 'three_ops_wrapped'
    }
    list_of_inputs.append(copy.deepcopy(input_2))

    # Input 3: Non-square operators with explicit hints
    op3_1 = tf.linalg.LinearOperatorFullMatrix(np.random.rand(2, 3).astype(np.float32))
    op3_2 = tf.linalg.LinearOperatorFullMatrix(np.random.rand(4, 2).astype(np.float32))
    input_3 = {
        'operators': [ComparableLinearOperator(op3_1, 0), ComparableLinearOperator(op3_2, 1)],
        'is_non_singular': False, 'is_self_adjoint': False, 'is_positive_definite': False, 'is_square': False, 'name': 'non_square_wrapped'
    }
    list_of_inputs.append(copy.deepcopy(input_3))

    # Input 4: Batched operators
    op4_1 = tf.linalg.LinearOperatorFullMatrix(np.random.rand(2, 2, 2).astype(np.float32))
    op4_2 = tf.linalg.LinearOperatorFullMatrix(np.random.rand(2, 3, 3).astype(np.float32))
    input_4 = {
        'operators': [ComparableLinearOperator(op4_1, 0), ComparableLinearOperator(op4_2, 1)],
        'is_non_singular': None, 'is_self_adjoint': None, 'is_positive_definite': None, 'is_square': True, 'name': 'batched_wrapped'
    }
    list_of_inputs.append(copy.deepcopy(input_4))

    # Input 5: Broadcasting batch dimensions
    op5_1 = tf.linalg.LinearOperatorFullMatrix(np.random.rand(4, 1, 2, 3).astype(np.float32))
    op5_2 = tf.linalg.LinearOperatorFullMatrix(np.random.rand(1, 5, 4, 2).astype(np.float32))
    input_5 = {
        'operators': [ComparableLinearOperator(op5_1, 0), ComparableLinearOperator(op5_2, 1)],
        'is_non_singular': None, 'is_self_adjoint': None, 'is_positive_definite': None, 'is_square': False, 'name': 'broadcast_wrapped'
    }
    list_of_inputs.append(copy.deepcopy(input_5))

    # Input 6: Explicitly set hints to True
    op6_1 = tf.linalg.LinearOperatorFullMatrix(np.array([[2., 1.], [1., 2.]], dtype=np.float32))
    op6_2 = tf.linalg.LinearOperatorIdentity(num_rows=3, dtype=tf.float32)
    input_6 = {
        'operators': [ComparableLinearOperator(op6_1, 0), ComparableLinearOperator(op6_2, 1)],
        'is_non_singular': True, 'is_self_adjoint': True, 'is_positive_definite': True, 'is_square': True, 'name': 'hints_true_wrapped'
    }
    list_of_inputs.append(copy.deepcopy(input_6))

    # Input 7: Mixed `LinearOperator` types
    op7_1 = tf.linalg.LinearOperatorFullMatrix(np.random.rand(2, 2).astype(np.float32))
    op7_2 = tf.linalg.LinearOperatorDiag(np.array([1., 2., 3.], dtype=np.float32))
    input_7 = {
        'operators': [ComparableLinearOperator(op7_1, 0), ComparableLinearOperator(op7_2, 1)],
        'is_non_singular': None, 'is_self_adjoint': None, 'is_positive_definite': None, 'is_square': True, 'name': 'mixed_types_wrapped'
    }
    list_of_inputs.append(copy.deepcopy(input_7))

    # Input 8: float64 dtype
    op8_1 = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 2.], [3., 4.]], dtype=np.float64))
    op8_2 = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 0.], [2., 1.]], dtype=np.float64))
    input_8 = {
        'operators': [ComparableLinearOperator(op8_1, 0), ComparableLinearOperator(op8_2, 1)],
        'is_non_singular': None, 'is_self_adjoint': None, 'is_positive_definite': None, 'is_square': None, 'name': 'float64_wrapped'
    }
    list_of_inputs.append(copy.deepcopy(input_8))

    # Input 9: complex64 dtype
    op9_1 = tf.linalg.LinearOperatorFullMatrix(np.array([[1+1j, 2], [3, 4-2j]], dtype=np.complex64))
    op9_2 = tf.linalg.LinearOperatorFullMatrix(np.array([[1j, 0], [2-1j, 1]], dtype=np.complex64))
    input_9 = {
        'operators': [ComparableLinearOperator(op9_1, 0), ComparableLinearOperator(op9_2, 1)],
        'is_non_singular': None, 'is_self_adjoint': None, 'is_positive_definite': None, 'is_square': True, 'name': 'complex64_wrapped'
    }
    list_of_inputs.append(copy.deepcopy(input_9))

    # Input 10: Single operator in the list
    op10_1 = tf.linalg.LinearOperatorFullMatrix(np.array([[5., -1.], [-1., 5.]], dtype=np.float32))
    input_10 = {
        'operators': [ComparableLinearOperator(op10_1, 0)],
        'is_non_singular': True, 'is_self_adjoint': True, 'is_positive_definite': True, 'is_square': True, 'name': 'single_op_wrapped'
    }
    list_of_inputs.append(copy.deepcopy(input_10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.LinearOperatorKronecker"] = get_linear_operator_kronecker_inputs()

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
