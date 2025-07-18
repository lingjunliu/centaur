
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_tf_linalg_linearoperatoradjoint_inputs():
    """
    Generates a list of valid inputs for the tf.linalg.LinearOperatorAdjoint function.
    """
    list_of_inputs = []

    # This wrapper class satisfies two conflicting requirements:
    # 1. The TensorFlow API, which needs a `LinearOperator` instance.
    # 2. The analysis tool, which (based on the error) expects an array-like object
    #    with a `.size` attribute and compatibility with numpy functions like `np.min`/`np.max`.
    # It inherits from `LinearOperatorFullMatrix` and adds the necessary array-like features.
    class AnalysableLinearOperator(tf.linalg.LinearOperatorFullMatrix):
        def __init__(self, matrix, *args, **kwargs):
            # Store the numpy array for the analysis tool
            self._matrix_numpy = np.array(matrix)
            # Initialize the parent LinearOperator
            super().__init__(matrix, *args, **kwargs)

        @property
        def size(self):
            return self._matrix_numpy.size

        # This protocol allows numpy functions (np.min, np.max) to work on this object
        def __array__(self):
            return self._matrix_numpy

    # Input 1: Simple 2x2 real matrix
    op1 = AnalysableLinearOperator(np.array([[1., 2.], [3., 4.]], dtype=np.float32))
    input_dict_1 = {
        'operator': op1,
        'is_non_singular': True,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'simple_real_2x2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Simple 2x2 complex matrix
    op2 = AnalysableLinearOperator(np.array([[1-1j, 3.], [0., 1+1j]], dtype=np.complex64))
    input_dict_2 = {
        'operator': op2,
        'is_non_singular': True,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'simple_complex_2x2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Non-square real matrix (3x2)
    op3 = AnalysableLinearOperator(np.array([[1., 2.], [3., 4.], [5., 6.]], dtype=np.float32))
    input_dict_3 = {
        'operator': op3,
        'is_non_singular': False,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': False,
        'name': 'non_square_real_3x2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Batched real self-adjoint operator
    op4 = AnalysableLinearOperator(np.array([[[4., 1.], [1., 3.]], [[5., 2.], [2., 5.]]], dtype=np.float32))
    input_dict_4 = {
        'operator': op4,
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'batched_real_self_adjoint'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Singular real matrix
    op5 = AnalysableLinearOperator(np.array([[1., 2.], [2., 4.]], dtype=np.float64))
    input_dict_5 = {
        'operator': op5,
        'is_non_singular': False,
        'is_self_adjoint': True,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'singular_real_matrix'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Identity matrix (real, float64)
    op6 = AnalysableLinearOperator(np.eye(3, dtype=np.float64))
    input_dict_6 = {
        'operator': op6,
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'identity_operator_float64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Zero matrix (non-square)
    op7 = AnalysableLinearOperator(np.zeros((4, 2), dtype=np.float32))
    input_dict_7 = {
        'operator': op7,
        'is_non_singular': False,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': False,
        'name': 'zero_operator_4x2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Matrix with negative values
    op8 = AnalysableLinearOperator(np.array([[-1., -2.], [3., -4.]], dtype=np.float32))
    input_dict_8 = {
        'operator': op8,
        'is_non_singular': True,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'negative_values_matrix'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Batched complex Hermitian (self-adjoint) operator
    op9 = AnalysableLinearOperator(np.array([[[2., 1.j], [-1.j, 2.]], [[3., 2.+1.j], [2.-1.j, 3.]]], dtype=np.complex64))
    input_dict_9 = {
        'operator': op9,
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'batched_complex_hermitian'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Batched random complex non-hermitian matrix
    op10 = AnalysableLinearOperator((np.random.rand(2, 4, 4) + 1j * np.random.rand(2, 4, 4)).astype(np.complex128))
    input_dict_10 = {
        'operator': op10,
        'is_non_singular': True,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'large_batched_complex'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.linalg.LinearOperatorAdjoint"] = get_tf_linalg_linearoperatoradjoint_inputs()

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
