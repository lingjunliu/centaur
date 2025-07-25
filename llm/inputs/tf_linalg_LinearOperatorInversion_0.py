
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

# The testing framework expects a `.size` attribute on the 'tensor' input,
# but tf.linalg.LinearOperator instances do not have one. This custom class
# inherits from LinearOperatorFullMatrix and adds the required attribute
# to satisfy the test harness while still being a valid input for the API.
class _HarnessCompatibleLinearOperator(tf.linalg.LinearOperatorFullMatrix):
    @property
    def size(self):
        """Returns the total number of elements, for test harness compatibility."""
        return tf.reduce_prod(self.shape).numpy()

def tf_linalg_linearoperatorinversion_inputs():
    """
    Generates a list of valid inputs for tf.linalg.LinearOperatorInversion.
    """
    list_of_inputs = []

    # Input 1: Basic 2x2 Identity Operator
    op1_matrix = np.array([[1., 0.], [0., 1.]], dtype=np.float32)
    op1 = _HarnessCompatibleLinearOperator(tf.constant(op1_matrix))
    input_dict1 = {
        'operator': op1,
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'identity_inversion'
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2x2 Diagonal Operator from documentation
    op2_matrix = np.array([[1., 0.], [0., 2.]], dtype=np.float32)
    op2 = _HarnessCompatibleLinearOperator(tf.constant(op2_matrix))
    input_dict2 = {
        'operator': op2,
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'diagonal_inversion'
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3x3 Symmetric Positive-Definite Operator
    op3_matrix = np.array([[4., 1., 1.], [1., 3., -1.], [1., -1., 2.]], dtype=np.float32)
    op3 = _HarnessCompatibleLinearOperator(tf.constant(op3_matrix))
    input_dict3 = {
        'operator': op3,
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'spd_3x3_inversion'
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 2x2 Non-Symmetric Operator with specified False hints
    op4_matrix = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    op4 = _HarnessCompatibleLinearOperator(tf.constant(op4_matrix))
    input_dict4 = {
        'operator': op4,
        'is_non_singular': True,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'general_2x2_inversion'
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Operator with all hints as None (default)
    op5_matrix = np.array([[5., 1.], [1., 3.]], dtype=np.float32)
    op5 = _HarnessCompatibleLinearOperator(tf.constant(op5_matrix))
    input_dict5 = {
        'operator': op5,
        'is_non_singular': None,
        'is_self_adjoint': None,
        'is_positive_definite': None,
        'is_square': None,
        'name': 'none_hints_inversion'
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Batched Operator (2, 3, 3)
    op6_matrix = np.array([
        [[2., 1., 0.], [1., 2., 1.], [0., 1., 2.]],
        [[3., 0., 0.], [0., 4., 0.], [0., 0., 5.]]
    ], dtype=np.float32)
    op6 = _HarnessCompatibleLinearOperator(tf.constant(op6_matrix))
    input_dict6 = {
        'operator': op6,
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'batch_inversion'
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Float64 dtype operator
    op7_matrix = np.array([[10., 1.], [1., 10.]], dtype=np.float64)
    op7 = _HarnessCompatibleLinearOperator(tf.constant(op7_matrix))
    input_dict7 = {
        'operator': op7,
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'float64_inversion'
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Self-Adjoint but not Positive-Definite Operator
    op8_matrix = np.array([[1., 2.], [2., -1.]], dtype=np.float32)
    op8 = _HarnessCompatibleLinearOperator(tf.constant(op8_matrix))
    input_dict8 = {
        'operator': op8,
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'self_adjoint_not_pd_inversion'
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Complex64 dtype operator
    op9_matrix = np.array([[1.+1.j, 2.+0.j], [0.+1.j, 3.-2.j]], dtype=np.complex64)
    op9 = _HarnessCompatibleLinearOperator(tf.constant(op9_matrix))
    input_dict9 = {
        'operator': op9,
        'is_non_singular': True,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'complex_inversion'
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    # Input 10: Larger 4x4 Operator
    op10_matrix = np.random.rand(4, 4).astype(np.float32) + np.eye(4, dtype=np.float32) * 5
    op10 = _HarnessCompatibleLinearOperator(tf.constant(op10_matrix))
    input_dict10 = {
        'operator': op10,
        'is_non_singular': True,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'large_4x4_inversion'
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    # Input 11: A singular matrix with a 'is_non_singular=True' hint
    op11_matrix = np.array([[1., 2.], [1., 2.]], dtype=np.float32)
    op11 = _HarnessCompatibleLinearOperator(tf.constant(op11_matrix))
    input_dict11 = {
        'operator': op11,
        'is_non_singular': True,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'singular_inversion_with_lie'
    }
    list_of_inputs.append(copy.deepcopy(input_dict11))
    
    return list_of_inputs

generated_inputs["tf.linalg.LinearOperatorInversion"] = tf_linalg_linearoperatorinversion_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.linalg.LinearOperatorInversion' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorInversion'.")

check_valid('tf.linalg.LinearOperatorInversion', generated_inputs['tf.linalg.LinearOperatorInversion'], lib="tf", suffix=0)
