
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_linearoperator_kronecker_inputs():
    list_of_inputs = []

    # Input 1
    operator_1 = np.array([[1., 2.], [3., 4.]])
    operator_2 = np.array([[1., 0.], [2., 1.]])
    operators = [tf.linalg.LinearOperatorFullMatrix(operator_1), tf.linalg.LinearOperatorFullMatrix(operator_2)]
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = None
    name = "kronecker_product_1"

    input_dict = {
        "operators": operators,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    operator_1 = np.array([[1., 0.], [0., 1.]])
    operator_2 = np.array([[0., 1.], [1., 0.]])
    operators = [tf.linalg.LinearOperatorFullMatrix(operator_1), tf.linalg.LinearOperatorFullMatrix(operator_2)]
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "kronecker_product_2"

    input_dict = {
        "operators": operators,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    matrix_1 = np.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]])
    matrix_2 = np.array([[[1., 0.], [2., 1.]], [[3., 4.], [5., 6.]]])

    operators = [
        tf.linalg.LinearOperatorFullMatrix(matrix_1),
        tf.linalg.LinearOperatorFullMatrix(matrix_2)
    ]
    is_non_singular = False
    is_self_adjoint = False
    is_positive_definite = None
    is_square = None
    name = "kronecker_product_3"

    input_dict = {
        "operators": operators,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 4: More operators
    matrix_1 = np.array([[1., 2.], [3., 4.]])
    matrix_2 = np.array([[1., 0.], [2., 1.]])
    operators = [
        tf.linalg.LinearOperatorFullMatrix(matrix_1),
        tf.linalg.LinearOperatorFullMatrix(matrix_2),
    ]
    is_non_singular = True
    is_self_adjoint = None
    is_positive_definite = None
    is_square = True
    name = "kronecker_product_4"

    input_dict = {
        "operators": operators,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Identity and Zero operators
    operators = [
        tf.linalg.LinearOperatorIdentity(num_rows=2),
        tf.linalg.LinearOperatorFullMatrix(np.zeros((2, 2)))
    ]
    is_non_singular = False
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "kronecker_product_5"

    input_dict = {
        "operators": operators,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Batch Operators
    matrix_1 = np.array([[[1., 0.], [0., 1.]], [[0., 1.], [1., 0.]]])
    matrix_2 = np.array([[[2., 0.], [0., 2.]], [[3., 0.], [0., 3.]]])

    operators = [
        tf.linalg.LinearOperatorFullMatrix(matrix_1),
        tf.linalg.LinearOperatorFullMatrix(matrix_2)
    ]
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "kronecker_product_6"

    input_dict = {
        "operators": operators,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: one element matrix
    matrix_1 = np.array([[5.]])
    matrix_2 = np.array([[2.]])

    operators = [
        tf.linalg.LinearOperatorFullMatrix(matrix_1, is_square = True),
        tf.linalg.LinearOperatorFullMatrix(matrix_2)
    ]

    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "kronecker_product_7"

    input_dict = {
        "operators": operators,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: None values for hints

    matrix_1 = np.array([[1., 2.], [3., 4.]])
    matrix_2 = np.array([[1., 0.], [2., 1.]])

    operators = [
        tf.linalg.LinearOperatorFullMatrix(matrix_1),
        tf.linalg.LinearOperatorFullMatrix(matrix_2)
    ]

    input_dict = {
        "operators": operators,
        "is_non_singular": None,
        "is_self_adjoint": None,
        "is_positive_definite": None,
        "is_square": None,
        "name": 'test_none'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.LinearOperatorKronecker"] = tf_linalg_linearoperator_kronecker_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.LinearOperatorKronecker' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorKronecker'.")

check_valid('tf.linalg.LinearOperatorKronecker', generated_inputs['tf.linalg.LinearOperatorKronecker'], lib="tf", suffix=0)
