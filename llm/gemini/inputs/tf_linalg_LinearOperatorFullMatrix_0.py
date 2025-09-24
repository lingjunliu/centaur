
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_linearoperatorfullmatrix_inputs():
    list_of_inputs = []

    # Input 1: Simple 2x2 matrix
    matrix = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    input_dict = {"matrix": matrix, "is_non_singular": True, "is_self_adjoint": False, "is_positive_definite": False, "is_square": True, "name": "simple_matrix"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Batch of matrices
    matrix = np.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]], dtype=np.float32)
    input_dict = {"matrix": matrix, "is_non_singular": True, "is_self_adjoint": False, "is_positive_definite": False, "is_square": True, "name": "batch_matrix"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Non-square matrix
    matrix = np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32)
    input_dict = {"matrix": matrix, "is_non_singular": None, "is_self_adjoint": None, "is_positive_definite": None, "is_square": False, "name": "non_square_matrix"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1x1 matrix
    matrix = np.array([[5.]], dtype=np.float32)
    input_dict = {"matrix": matrix, "is_non_singular": True, "is_self_adjoint": True, "is_positive_definite": True, "is_square": True, "name": "one_by_one"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger matrix
    matrix = np.random.rand(5, 5).astype(np.float32)
    input_dict = {"matrix": matrix, "is_non_singular": None, "is_self_adjoint": None, "is_positive_definite": None, "is_square": True, "name": "large_matrix"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Complex matrix
    matrix = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex64)
    input_dict = {"matrix": matrix, "is_non_singular": None, "is_self_adjoint": None, "is_positive_definite": None, "is_square": True, "name": "complex_matrix"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Batch of complex matrices
    matrix = np.array([[[1+1j, 2+2j], [3+3j, 4+4j]], [[5+5j, 6+6j], [7+7j, 8+8j]]], dtype=np.complex64)
    input_dict = {"matrix": matrix, "is_non_singular": None, "is_self_adjoint": None, "is_positive_definite": None, "is_square": True, "name": "batch_complex_matrix"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Matrix with negative values
    matrix = np.array([[-1., 2.], [3., -4.]], dtype=np.float32)
    input_dict = {"matrix": matrix, "is_non_singular": None, "is_self_adjoint": False, "is_positive_definite": False, "is_square": True, "name": "negative_matrix"}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9: 3D Tensor representing a batch of non-square matrices
    matrix = np.random.rand(2, 3, 4).astype(np.float32)
    input_dict = {"matrix": matrix, "is_non_singular": None, "is_self_adjoint": None, "is_positive_definite": None, "is_square": False, "name": "batch_non_square"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Matrix that is symmetric
    matrix = np.array([[1., 2.], [2., 4.]], dtype=np.float32)
    input_dict = {"matrix": matrix, "is_non_singular": None, "is_self_adjoint": True, "is_positive_definite": None, "is_square": True, "name": "symmetric_matrix"}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.LinearOperatorFullMatrix"] = tf_linalg_linearoperatorfullmatrix_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.LinearOperatorFullMatrix' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorFullMatrix'.")

check_valid('tf.linalg.LinearOperatorFullMatrix', generated_inputs['tf.linalg.LinearOperatorFullMatrix'], lib="tf", suffix=0)
