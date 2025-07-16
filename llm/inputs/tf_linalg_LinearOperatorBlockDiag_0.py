
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_linearoperatorblockdiag_inputs():
    list_of_inputs = []

    # Input 1: Basic case with two 2x2 matrices
    matrix1 = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    matrix2 = np.array([[5., 6.], [7., 8.]], dtype=np.float32)
    operator1 = tf.linalg.LinearOperatorFullMatrix(matrix1)
    operator2 = tf.linalg.LinearOperatorFullMatrix(matrix2)
    operators = [operator1, operator2]
    input_dict = {"operators": operators, "is_non_singular": True, "is_self_adjoint": False, "is_positive_definite": False, "is_square": True, "name": "block_diag_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2:  One 3x3 and one 2x2, square False
    matrix1 = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], dtype=np.float32)
    matrix2 = np.array([[1., 0.], [0., 1.]], dtype=np.float32)
    operator1 = tf.linalg.LinearOperatorFullMatrix(matrix1)
    operator2 = tf.linalg.LinearOperatorFullMatrix(matrix2)
    operators = [operator1, operator2]
    input_dict = {"operators": operators, "is_non_singular": None, "is_self_adjoint": None, "is_positive_definite": None, "is_square": True, "name": "block_diag_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3:  Rectangular matrices
    matrix1 = np.array([[1., 2.], [3., 4.], [5., 6.]], dtype=np.float32)
    matrix2 = np.array([[7., 8., 9.]], dtype=np.float32)
    operator1 = tf.linalg.LinearOperatorFullMatrix(matrix1)
    operator2 = tf.linalg.LinearOperatorFullMatrix(matrix2)
    operators = [operator1, operator2]
    input_dict = {"operators": operators, "is_non_singular": False, "is_self_adjoint": False, "is_positive_definite": False, "is_square": False, "name": "block_diag_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4:  Batched operators
    matrix1 = np.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]], dtype=np.float32)
    matrix2 = np.array([[[9., 10.], [11., 12.]], [[13., 14.], [15., 16.]]], dtype=np.float32)
    operator1 = tf.linalg.LinearOperatorFullMatrix(matrix1)
    operator2 = tf.linalg.LinearOperatorFullMatrix(matrix2)
    operators = [operator1, operator2]
    input_dict = {"operators": operators, "is_non_singular": None, "is_self_adjoint": None, "is_positive_definite": None, "is_square": True, "name": "block_diag_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Single operator
    matrix1 = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    operator1 = tf.linalg.LinearOperatorFullMatrix(matrix1)
    operators = [operator1]
    input_dict = {"operators": operators, "is_non_singular": True, "is_self_adjoint": False, "is_positive_definite": False, "is_square": True, "name": "block_diag_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: is_positive_definite = True, self_adjoint = True
    matrix1 = np.array([[2., 1.], [1., 2.]], dtype=np.float32)
    operator1 = tf.linalg.LinearOperatorFullMatrix(matrix1)
    operators = [operator1]
    input_dict = {"operators": operators, "is_non_singular": True, "is_self_adjoint": True, "is_positive_definite": True, "is_square": True, "name": "block_diag_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: is_non_singular = False
    matrix1 = np.array([[0., 0.], [0., 0.]], dtype=np.float32)
    operator1 = tf.linalg.LinearOperatorFullMatrix(matrix1)
    operators = [operator1]
    input_dict = {"operators": operators, "is_non_singular": False, "is_self_adjoint": False, "is_positive_definite": False, "is_square": True, "name": "block_diag_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: different dtypes
    matrix1 = np.array([[1., 2.], [3., 4.]], dtype=np.float64)
    matrix2 = np.array([[5., 6.], [7., 8.]], dtype=np.complex64)
    operator1 = tf.linalg.LinearOperatorFullMatrix(matrix1)
    operator2 = tf.linalg.LinearOperatorFullMatrix(matrix2)
    operators = [operator1, operator2]
    input_dict = {"operators": operators, "is_non_singular": None, "is_self_adjoint": None, "is_positive_definite": None, "is_square": True, "name": "block_diag_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty operators list
    operators = []
    input_dict = {"operators": operators, "is_non_singular": None, "is_self_adjoint": None, "is_positive_definite": None, "is_square": True, "name": "block_diag_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: is_square=False, rectangular
    matrix1 = np.array([[1., 2.], [3., 4.], [5., 6.]], dtype=np.float32)
    operator1 = tf.linalg.LinearOperatorFullMatrix(matrix1)
    operators = [operator1]
    input_dict = {"operators": operators, "is_non_singular": False, "is_self_adjoint": False, "is_positive_definite": False, "is_square": False, "name": "block_diag_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.LinearOperatorBlockDiag"] = tf_linalg_linearoperatorblockdiag_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.LinearOperatorBlockDiag' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorBlockDiag'.")

check_valid('tf.linalg.LinearOperatorBlockDiag', generated_inputs['tf.linalg.LinearOperatorBlockDiag'], lib="tf", suffix=0)
