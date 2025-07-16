
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_linearoperatoradjoint_inputs():
    list_of_inputs = []

    def create_linear_operator(matrix, is_non_singular=None, is_self_adjoint=None, is_positive_definite=None, is_square=None, name=None):
        operator = tf.linalg.LinearOperatorFullMatrix(matrix)
        return {"operator": operator, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name, "matrix_shape": matrix.shape}

    # Input 1: Simple square matrix
    matrix1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    input_dict1 = create_linear_operator(matrix1, is_non_singular=True, is_self_adjoint=False, is_positive_definite=False, is_square=True, name="adjoint_op1")
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Non-square matrix
    matrix2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    input_dict2 = create_linear_operator(matrix2, is_non_singular=None, is_self_adjoint=None, is_positive_definite=None, is_square=False, name="adjoint_op2")
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Complex matrix
    matrix3 = np.array([[1.0 + 1j, 2.0], [3.0, 4.0 - 1j]])
    input_dict3 = create_linear_operator(matrix3, is_non_singular=False, is_self_adjoint=None, is_positive_definite=None, is_square=True, name="adjoint_op3")
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Matrix with zeros
    matrix4 = np.array([[0.0, 1.0], [1.0, 0.0]])
    input_dict4 = create_linear_operator(matrix4, is_non_singular=False, is_self_adjoint=True, is_positive_definite=False, is_square=True, name="adjoint_op4")
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Batch matrix
    matrix5 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    input_dict5 = create_linear_operator(matrix5, is_non_singular=True, is_self_adjoint=False, is_positive_definite=False, is_square=True, name="adjoint_op5")
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Identity matrix
    matrix6 = np.eye(3)
    input_dict6 = create_linear_operator(matrix6, is_non_singular=True, is_self_adjoint=True, is_positive_definite=True, is_square=True, name="adjoint_op6")
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Negative values
    matrix7 = np.array([[-1.0, 0.0], [0.0, -1.0]])
    input_dict7 = create_linear_operator(matrix7, is_non_singular=True, is_self_adjoint=True, is_positive_definite=False, is_square=True, name="adjoint_op7")
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: None flags
    matrix8 = np.array([[1.0, 0.0], [0.0, 1.0]])
    input_dict8 = create_linear_operator(matrix8, is_non_singular=None, is_self_adjoint=None, is_positive_definite=None, is_square=True, name="adjoint_op8")
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    # Input 9: Rectangular matrix
    matrix9 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    input_dict9 = create_linear_operator(matrix9, is_non_singular=False, is_self_adjoint=None, is_positive_definite=None, is_square=False, name="adjoint_op9")
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Zero matrix
    matrix10 = np.array([[0.0, 0.0], [0.0, 0.0]])
    input_dict10 = create_linear_operator(matrix10, is_non_singular=False, is_self_adjoint=True, is_positive_definite=False, is_square=True, name="adjoint_op10")
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.LinearOperatorAdjoint"] = tf_linalg_linearoperatoradjoint_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.LinearOperatorAdjoint' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorAdjoint'.")

check_valid('tf.linalg.LinearOperatorAdjoint', generated_inputs['tf.linalg.LinearOperatorAdjoint'], lib="tf", suffix=0)
