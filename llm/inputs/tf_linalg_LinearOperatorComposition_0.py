
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_LinearOperatorComposition_inputs():
    list_of_inputs = []

    # Input 1
    matrix1 = np.array([[1., 2.], [3., 4.]])
    matrix2 = np.array([[5., 6.], [7., 8.]])
    operator1 = tf.linalg.LinearOperatorFullMatrix(matrix1)
    operator2 = tf.linalg.LinearOperatorFullMatrix(matrix2)
    operators = [operator1, operator2]
    is_non_singular = True
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "composition_1"

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
    operator1 = tf.linalg.LinearOperatorScaledIdentity(num_rows=3, multiplier=2.0)
    matrix2 = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    operator2 = tf.linalg.LinearOperatorFullMatrix(matrix2)
    operators = [operator1, operator2]
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = None
    name = "composition_2"

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
    matrix1 = np.random.rand(2, 3, 4, 5)
    matrix2 = np.random.rand(2, 3, 5, 6)
    operator1 = tf.linalg.LinearOperatorFullMatrix(matrix1)
    operator2 = tf.linalg.LinearOperatorFullMatrix(matrix2)
    operators = [operator1, operator2]
    is_non_singular = False
    is_self_adjoint = False
    is_positive_definite = False
    is_square = False
    name = "composition_3"

    input_dict = {
        "operators": operators,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 4
    matrix1 = np.random.rand(4, 4)
    matrix2 = np.random.rand(4, 4)
    operator1 = tf.linalg.LinearOperatorFullMatrix(matrix1)
    operator2 = tf.linalg.LinearOperatorFullMatrix(matrix2)
    operators = [operator1, operator2]
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "composition_4"

    input_dict = {
        "operators": operators,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    operator1 = tf.linalg.LinearOperatorScaledIdentity(num_rows=2, multiplier=1.0)
    operators = [operator1]
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = True
    name = "composition_5"

    input_dict = {
        "operators": operators,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    matrix1 = np.random.rand(2, 3, 4, 5)
    matrix2 = np.random.rand(2, 3, 5, 6)
    operator1 = tf.linalg.LinearOperatorFullMatrix(matrix1)
    operator2 = tf.linalg.LinearOperatorFullMatrix(matrix2)
    operators = [operator1, operator2]
    is_non_singular = False
    is_self_adjoint = False
    is_positive_definite = False
    is_square = False
    name = "composition_6"

    input_dict = {
        "operators": operators,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    operator1 = tf.linalg.LinearOperatorScaledIdentity(num_rows=3, multiplier=-2.0)
    matrix2 = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    operator2 = tf.linalg.LinearOperatorFullMatrix(matrix2)
    operators = [operator1, operator2]
    is_non_singular = False
    is_self_adjoint = None
    is_positive_definite = False
    is_square = True
    name = "composition_7"

    input_dict = {
        "operators": operators,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    matrix1 = np.random.rand(2, 2)
    matrix2 = np.random.rand(2, 2)
    operator1 = tf.linalg.LinearOperatorFullMatrix(matrix1)
    operator2 = tf.linalg.LinearOperatorFullMatrix(matrix2)
    operators = [operator1, operator2]
    is_non_singular = True
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "composition_8"

    input_dict = {
        "operators": operators,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    operator1 = tf.linalg.LinearOperatorIdentity(num_rows=4)
    matrix2 = np.eye(4)
    operator2 = tf.linalg.LinearOperatorFullMatrix(matrix2)
    operators = [operator1, operator2]
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "composition_9"

    input_dict = {
        "operators": operators,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    matrix1 = np.random.rand(1, 1)
    matrix2 = np.random.rand(1, 1)
    operator1 = tf.linalg.LinearOperatorFullMatrix(matrix1)
    operator2 = tf.linalg.LinearOperatorFullMatrix(matrix2)
    operators = [operator1, operator2]
    is_non_singular = False
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "composition_10"

    input_dict = {
        "operators": operators,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.linalg.LinearOperatorComposition"] = tf_linalg_LinearOperatorComposition_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.LinearOperatorComposition' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorComposition'.")

check_valid('tf.linalg.LinearOperatorComposition', generated_inputs['tf.linalg.LinearOperatorComposition'], lib="tf", suffix=0)
