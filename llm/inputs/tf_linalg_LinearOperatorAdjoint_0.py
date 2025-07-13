
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_linearoperatoradjoint_inputs():
    list_of_inputs = []

    # Input 1
    matrix = np.array([[1., 2.], [3., 4.]])
    operator = tf.linalg.LinearOperatorFullMatrix(matrix)
    is_non_singular = True
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "adjoint_op_1"
    input_dict = {"operator": matrix, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    matrix = np.array([[1. + 1j, 2.], [3., 4. - 1j]])
    operator = tf.linalg.LinearOperatorFullMatrix(matrix)
    is_non_singular = False
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "adjoint_op_2"
    input_dict = {"operator": matrix, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    matrix = np.array([[5., 0.], [0., 5.]])
    operator = tf.linalg.LinearOperatorFullMatrix(matrix)
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "adjoint_op_3"
    input_dict = {"operator": matrix, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    matrix = np.array([[1., 0.], [0., -1.]])
    operator = tf.linalg.LinearOperatorFullMatrix(matrix)
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "adjoint_op_4"
    input_dict = {"operator": matrix, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    matrix = np.eye(3)
    operator = tf.linalg.LinearOperatorFullMatrix(matrix)
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "adjoint_op_5"
    input_dict = {"operator": matrix, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6
    matrix = np.array([[0., 1.], [1., 0.]])
    operator = tf.linalg.LinearOperatorFullMatrix(matrix)
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "adjoint_op_6"
    input_dict = {"operator": matrix, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    matrix = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    operator = tf.linalg.LinearOperatorFullMatrix(matrix)
    is_non_singular = False
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "adjoint_op_7"
    input_dict = {"operator": matrix, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
    operator = tf.linalg.LinearOperatorFullMatrix(matrix)
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "adjoint_op_8"
    input_dict = {"operator": matrix, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    matrix = np.array([[2.0, 1.0], [1.0, 2.0]])
    operator = tf.linalg.LinearOperatorFullMatrix(matrix)
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "adjoint_op_9"
    input_dict = {"operator": matrix, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    matrix = np.array([[1.0, 0.0], [0.0, -1.0]])
    operator = tf.linalg.LinearOperatorFullMatrix(matrix)
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "adjoint_op_10"
    input_dict = {"operator": matrix, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

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
