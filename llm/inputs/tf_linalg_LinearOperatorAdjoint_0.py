
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

# Helper class to satisfy the testing framework's pre-check on the 'operator'
# argument, which expects a `.size` attribute, while also satisfying the API's
# requirement for a tf.linalg.LinearOperator instance.
class _SizedLinearOperatorFullMatrix(tf.linalg.LinearOperatorFullMatrix):
    """A LinearOperatorFullMatrix that has a .size property."""
    def __init__(self, matrix, **kwargs):
        self._matrix_tensor = tf.convert_to_tensor(matrix)
        super().__init__(self._matrix_tensor, **kwargs)

    @property
    def size(self):
        """The .size attribute required by the testing framework."""
        if hasattr(self._matrix_tensor, 'numpy'):
            return self._matrix_tensor.numpy().size
        return tf.size(self._matrix_tensor)

def tf_linalg_linearoperatoradjoint_inputs():
    """
    Generates a list of valid inputs for the tf.linalg.LinearOperatorAdjoint function.
    """
    list_of_inputs = []

    # Input 1: Basic 2x2 real, non-singular operator
    operator_1 = _SizedLinearOperatorFullMatrix(
        np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    )
    input_dict_1 = {
        'operator': operator_1,
        'is_non_singular': True,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'real_2x2_nonsingular'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Basic 2x2 complex, non-singular operator (from docs)
    operator_2 = _SizedLinearOperatorFullMatrix(
        np.array([[1. - 1.j, 3.], [0., 1. + 1.j]], dtype=np.complex64)
    )
    input_dict_2 = {
        'operator': operator_2,
        'is_non_singular': True,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'complex_2x2_from_docs'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 2x2 singular real operator
    operator_3 = _SizedLinearOperatorFullMatrix(
        np.array([[1., 2.], [2., 4.]], dtype=np.float64),
        is_self_adjoint=True
    )
    input_dict_3 = {
        'operator': operator_3,
        'is_non_singular': False,
        'is_self_adjoint': True,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'real_2x2_singular'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 2x2 complex self-adjoint and positive-definite operator
    operator_4 = _SizedLinearOperatorFullMatrix(
        np.array([[2., 1. + 1.j], [1. - 1.j, 3.]], dtype=np.complex128),
        is_self_adjoint=True, is_positive_definite=True
    )
    input_dict_4 = {
        'operator': operator_4,
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'complex_2x2_self_adjoint_pd'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Non-square (2x3) operator with float32
    operator_5 = _SizedLinearOperatorFullMatrix(
        np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32)
    )
    input_dict_5 = {
        'operator': operator_5,
        'is_non_singular': None,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': False,
        'name': 'nonsquare_2x3_operator'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Non-square (3x2) operator with float64
    operator_6 = _SizedLinearOperatorFullMatrix(
        np.array([[1., 2.], [3., 4.], [5., 6.]], dtype=np.float64)
    )
    input_dict_6 = {
        'operator': operator_6,
        'is_non_singular': None,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': False,
        'name': 'nonsquare_3x2_operator'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Input 7: 3x3 positive-definite operator
    operator_7 = _SizedLinearOperatorFullMatrix(
        np.array([[4., 1., 1.], [1., 3., -1.], [1., -1., 2.]], dtype=np.float32),
        is_self_adjoint=True, is_positive_definite=True
    )
    input_dict_7 = {
        'operator': operator_7,
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'real_3x3_pd'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Batch operator (2, 2, 3)
    operator_8 = _SizedLinearOperatorFullMatrix(
        np.random.rand(2, 2, 3).astype(np.float32)
    )
    input_dict_8 = {
        'operator': operator_8,
        'is_non_singular': None,
        'is_self_adjoint': None,
        'is_positive_definite': None,
        'is_square': False,
        'name': 'batch_operator_nonsquare'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Operator with all boolean hints as None (default behavior)
    operator_9 = _SizedLinearOperatorFullMatrix(
        np.array([[1., 0.], [0., -1.]], dtype=np.float64)
    )
    input_dict_9 = {
        'operator': operator_9,
        'is_non_singular': None,
        'is_self_adjoint': None,
        'is_positive_definite': None,
        'is_square': None,
        'name': 'all_hints_none'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: 1x1 operator
    operator_10 = _SizedLinearOperatorFullMatrix(
        np.array([[-10.]], dtype=np.float32)
    )
    input_dict_10 = {
        'operator': operator_10,
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'scalar_operator'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

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
