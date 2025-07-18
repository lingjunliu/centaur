
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

class CustomLinearOperator(tf.linalg.LinearOperatorFullMatrix):
    """
    A wrapper for LinearOperatorFullMatrix to make it compatible with a test
    harness that requires a `.size` attribute for analysis, while the API
    itself requires a LinearOperator instance.
    """
    @property
    def size(self):
        return tf.size(self.to_dense()).numpy()

    def __deepcopy__(self, memo):
        # Create a new instance of the class with a deep copy of the matrix
        cls = self.__class__
        result = cls(
            matrix=copy.deepcopy(self.to_dense().numpy()),
            is_non_singular=self.is_non_singular,
            is_self_adjoint=self.is_self_adjoint,
            is_positive_definite=self.is_positive_definite,
            is_square=self.is_square,
            name=self.name + "_copy"
        )
        memo[id(self)] = result
        return result

def tf_linalg_linearoperatoradjoint_inputs():
    """
    Generates a list of valid inputs for tf.linalg.LinearOperatorAdjoint.
    """
    list_of_inputs = []

    # Input 1: Basic 2x2 real matrix (float32), no hints
    operator1 = CustomLinearOperator(np.array([[1., 2.], [3., 4.]], dtype=np.float32))
    input_dict1 = {
        'operator': operator1,
        'is_non_singular': None,
        'is_self_adjoint': None,
        'is_positive_definite': None,
        'is_square': None,
        'name': 'real_2x2_no_hints'
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Basic 2x2 complex matrix (complex64), from documentation
    operator2 = CustomLinearOperator(np.array([[1 - 1j, 3.], [0., 1. + 1j]], dtype=np.complex64))
    input_dict2 = {
        'operator': operator2,
        'is_non_singular': None,
        'is_self_adjoint': None,
        'is_positive_definite': None,
        'is_square': None,
        'name': 'complex_2x2_from_doc'
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Non-square 2x3 real matrix (float64)
    operator3 = CustomLinearOperator(np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float64))
    input_dict3 = {
        'operator': operator3,
        'is_non_singular': None,
        'is_self_adjoint': None,
        'is_positive_definite': None,
        'is_square': False,
        'name': 'real_2x3_nonsquare'
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Non-square 3x2 complex matrix (complex128)
    operator4 = CustomLinearOperator(np.array([[1.+2.j, 3.-1.j], [0., 5.j], [4., -1.+1.j]], dtype=np.complex128))
    input_dict4 = {
        'operator': operator4,
        'is_non_singular': None,
        'is_self_adjoint': None,
        'is_positive_definite': None,
        'is_square': False,
        'name': 'complex_3x2_nonsquare'
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Self-adjoint (Hermitian) 3x3 matrix with hints
    matrix5 = np.array([[2., 2.+1.j, 4.-5.j], [2.-1.j, 3., 8.+2.j], [4.+5.j, 8.-2.j, -1.]], dtype=np.complex64)
    operator5 = CustomLinearOperator(matrix5)
    input_dict5 = {
        'operator': operator5,
        'is_non_singular': None,
        'is_self_adjoint': True,
        'is_positive_definite': None,
        'is_square': True,
        'name': 'hermitian_3x3_hinted'
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Positive-definite 2x2 matrix with all hints True
    operator6 = CustomLinearOperator(np.array([[2., -1.], [-1., 2.]], dtype=np.float32))
    input_dict6 = {
        'operator': operator6,
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'pos_def_2x2_all_hints'
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Singular 2x2 matrix with hints
    operator7 = CustomLinearOperator(np.array([[1., 1.], [1., 1.]], dtype=np.float32))
    input_dict7 = {
        'operator': operator7,
        'is_non_singular': False,
        'is_self_adjoint': True,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'singular_2x2_hinted'
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Larger 4x4 real matrix (float64)
    operator8 = CustomLinearOperator(np.arange(16, dtype=np.float64).reshape(4, 4))
    input_dict8 = {
        'operator': operator8,
        'is_non_singular': False,
        'is_self_adjoint': None,
        'is_positive_definite': None,
        'is_square': True,
        'name': 'real_4x4_large_singular'
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Batch of 3, 2x2 matrices
    operator9 = CustomLinearOperator(np.array([[[1., 0.], [0., 1.]], [[2., 1.], [1., 2.]], [[3., 0.], [1., 3.]]], dtype=np.float32))
    input_dict9 = {
        'operator': operator9,
        'is_non_singular': True,
        'is_self_adjoint': None,
        'is_positive_definite': None,
        'is_square': True,
        'name': 'batch_2x2_real'
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Batch of 2, 2x3 non-square matrices
    operator10 = CustomLinearOperator(np.arange(12, dtype=np.float32).reshape(2, 2, 3))
    input_dict10 = {
        'operator': operator10,
        'is_non_singular': None,
        'is_self_adjoint': None,
        'is_positive_definite': None,
        'is_square': False,
        'name': 'batch_nonsquare_2x3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.linalg.LinearOperatorAdjoint"] = tf_linalg_linearoperatoradjoint_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.linalg.LinearOperatorAdjoint' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorAdjoint'.")

check_valid('tf.linalg.LinearOperatorAdjoint', generated_inputs['tf.linalg.LinearOperatorAdjoint'], lib="tf", suffix=0)
