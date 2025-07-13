
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_linearoperator_kronecker_inputs():
    list_of_inputs = []

    # Input 1: Basic case with two 2x2 operators
    operator_1 = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 2.], [3., 4.]]))
    operator_2 = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 0.], [2., 1.]]))
    operators = [operator_1, operator_2]
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = None
    name = "basic_kronecker"

    input_dict = {
        "operators": operators,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: is_square=True
    operator_1 = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 2.], [3., 4.]]))
    operator_2 = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 0.], [2., 1.]]))
    operators = [operator_1, operator_2]
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = True
    name = "square_kronecker"

    input_dict = {
        "operators": operators,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: is_non_singular=False
    operator_1 = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 2.], [3., 4.]]))
    operator_2 = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 0.], [2., 1.]]))
    operators = [operator_1, operator_2]
    is_non_singular = False
    is_self_adjoint = None
    is_positive_definite = None
    is_square = None
    name = "non_singular_false_kronecker"

    input_dict = {
        "operators": operators,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: is_positive_definite = True
    operator_1 = tf.linalg.LinearOperatorFullMatrix(np.array([[2., 1.], [1., 2.]]))
    operator_2 = tf.linalg.LinearOperatorFullMatrix(np.array([[3., 1.], [1., 3.]]))

    operators = [operator_1, operator_2]
    is_non_singular = None
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "positive_definite_kronecker"
    input_dict = {
        "operators": operators,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: is_self_adjoint = True
    operator_1 = tf.linalg.LinearOperatorFullMatrix(np.array([[2., 1.], [1., 2.]]))
    operator_2 = tf.linalg.LinearOperatorFullMatrix(np.array([[3., 1.], [1., 3.]]))

    operators = [operator_1, operator_2]
    is_non_singular = None
    is_self_adjoint = True
    is_positive_definite = None
    is_square = True
    name = "self_adjoint_kronecker"
    input_dict = {
        "operators": operators,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty name
    operator_1 = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 2.], [3., 4.]]))
    operator_2 = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 0.], [2., 1.]]))
    operators = [operator_1, operator_2]
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = None
    name = ""

    input_dict = {
        "operators": operators,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Three 1x1 operators
    operator_1 = tf.linalg.LinearOperatorFullMatrix(np.array([[1.]]))
    operator_2 = tf.linalg.LinearOperatorFullMatrix(np.array([[2.]]))
    operator_3 = tf.linalg.LinearOperatorFullMatrix(np.array([[3.]]))
    operators = [operator_1, operator_2, operator_3]
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = None
    name = "three_1x1_operators_kronecker"

    input_dict = {
        "operators": operators,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Two 1x1 operators
    operator_1 = tf.linalg.LinearOperatorFullMatrix(np.array([[4.]]))
    operator_2 = tf.linalg.LinearOperatorFullMatrix(np.array([[5.]]))
    operators = [operator_1, operator_2]
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = None
    name = "two_1x1_operators_kronecker"

    input_dict = {
        "operators": operators,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: One operator
    operator_1 = tf.linalg.LinearOperatorFullMatrix(np.array([[6.,7.], [8., 9.]]))
    operators = [operator_1]
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = None
    name = "one_operator_kronecker"

    input_dict = {
        "operators": operators,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different shape operators
    operator_1 = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 2.], [3., 4.]]))
    operator_2 = tf.linalg.LinearOperatorFullMatrix(np.array([[5., 6., 7.], [8., 9., 10.], [11.,12.,13.]]))
    operators = [operator_1, operator_2]
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = None
    name = "different_shape_operators"

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
