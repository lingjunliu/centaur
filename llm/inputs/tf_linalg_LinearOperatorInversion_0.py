
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_linalg_linearoperatorinversion_inputs():
    """
    Generates a list of valid inputs for tf.linalg.LinearOperatorInversion.
    The 'operator' parameter must be an instance of tf.linalg.LinearOperator.
    To satisfy the testing framework which expects a '.size' attribute on
    'tensor' type inputs, we dynamically add a 'size' attribute to the
    LinearOperator instance after it is created.
    """
    list_of_inputs = []

    # Input 1: Basic 2x2 identity matrix, float32
    matrix1 = np.array([[1., 0.], [0., 1.]], dtype=np.float32)
    op1 = tf.linalg.LinearOperatorFullMatrix(matrix1)
    op1.size = matrix1.size
    input_dict_1 = {
        'operator': op1,
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'identity_inversion'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2x2 diagonal matrix, float64, with some hints as None
    matrix2 = np.array([[3., 0.], [0., 0.5]], dtype=np.float64)
    op2 = tf.linalg.LinearOperatorFullMatrix(matrix2)
    op2.size = matrix2.size
    input_dict_2 = {
        'operator': op2,
        'is_non_singular': True,
        'is_self_adjoint': None,
        'is_positive_definite': None,
        'is_square': True,
        'name': 'diagonal_inversion'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 3x3 symmetric positive-definite matrix, no name
    matrix3 = np.array([[2., 1., 0.], [1., 2., 1.], [0., 1., 2.]], dtype=np.float32)
    op3 = tf.linalg.LinearOperatorFullMatrix(matrix3)
    op3.size = matrix3.size
    input_dict_3 = {
        'operator': op3,
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 3x3 non-symmetric but non-singular matrix
    matrix4 = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 10.]], dtype=np.float32)
    op4 = tf.linalg.LinearOperatorFullMatrix(matrix4)
    op4.size = matrix4.size
    input_dict_4 = {
        'operator': op4,
        'is_non_singular': True,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'non_symmetric_inversion'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Symmetric, non-singular, but not positive-definite
    matrix5 = np.array([[1., 0.], [0., -1.]], dtype=np.float32)
    op5 = tf.linalg.LinearOperatorFullMatrix(matrix5)
    op5.size = matrix5.size
    input_dict_5 = {
        'operator': op5,
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'indefinite_inversion'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Batch of 2x2 matrices
    matrix6 = np.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]], dtype=np.float32)
    op6 = tf.linalg.LinearOperatorFullMatrix(matrix6)
    op6.size = matrix6.size
    input_dict_6 = {
        'operator': op6,
        'is_non_singular': True,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'batch_inversion'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: All boolean hints set to None (default)
    matrix7 = np.array([[10., 1.], [1., 10.]], dtype=np.float32)
    op7 = tf.linalg.LinearOperatorFullMatrix(matrix7)
    op7.size = matrix7.size
    input_dict_7 = {
        'operator': op7,
        'is_non_singular': None,
        'is_self_adjoint': None,
        'is_positive_definite': None,
        'is_square': None,
        'name': 'all_none_hints'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Batch of matrices with multiple batch dimensions
    matrix8 = np.stack([np.eye(3, dtype=np.float64) * (i + 1) for i in range(4)]).reshape(2, 2, 3, 3)
    op8 = tf.linalg.LinearOperatorFullMatrix(matrix8)
    op8.size = matrix8.size
    input_dict_8 = {
        'operator': op8,
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'multi_batch_dim_inversion'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Large 4x4 matrix
    matrix9 = np.array([[4, 1, 0, 0], [1, 4, 1, 0], [0, 1, 4, 1], [0, 0, 1, 4]], dtype=np.float32)
    op9 = tf.linalg.LinearOperatorFullMatrix(matrix9)
    op9.size = matrix9.size
    input_dict_9 = {
        'operator': op9,
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'large_tridiagonal_inversion'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Complex Hermitian positive-definite operator
    matrix10 = np.array([[2., 1. + 1.j], [1. - 1.j, 3.]], dtype=np.complex64)
    op10 = tf.linalg.LinearOperatorFullMatrix(matrix10)
    op10.size = matrix10.size
    input_dict_10 = {
        'operator': op10,
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'complex_hermitian_inversion'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Complex non-Hermitian operator
    matrix11 = np.array([[1., 2. + 1.j], [3. - 2.j, 4.]], dtype=np.complex128)
    op11 = tf.linalg.LinearOperatorFullMatrix(matrix11)
    op11.size = matrix11.size
    input_dict_11 = {
        'operator': op11,
        'is_non_singular': True,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'complex_nonhermitian_inversion'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

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
