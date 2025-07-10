
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_linearoperatorcomposition_inputs():
    list_of_inputs = []

    # Input 1
    op1 = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 2.], [3., 4.]]))
    operators = [op1]
    is_non_singular = True
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "test_composition_1"
    input_dict = {"operators": operators, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint,
                  "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    op1 = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 0.], [0., 1.]]))
    op2 = tf.linalg.LinearOperatorFullMatrix(np.array([[0., 1.], [1., 0.]]))
    operators = [op1, op2]
    is_non_singular = False
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "test_composition_2"
    input_dict = {"operators": operators, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint,
                  "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    matrix_45 = np.random.normal(size=[4, 5])
    op_45 = tf.linalg.LinearOperatorFullMatrix(matrix_45)
    matrix_56 = np.random.normal(size=[5, 6])
    op_56 = tf.linalg.LinearOperatorFullMatrix(matrix_56)
    operators = [op_45, op_56]
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = False
    name = "test_composition_3"
    input_dict = {"operators": operators, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint,
                  "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    matrix_2345 = np.random.normal(size=[2, 3, 4, 5])
    op_2345 = tf.linalg.LinearOperatorFullMatrix(matrix_2345)
    matrix_2356 = np.random.normal(size=[2, 3, 5, 6])
    op_2356 = tf.linalg.LinearOperatorFullMatrix(matrix_2356)
    operators = [op_2345, op_2356]
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = False
    name = "test_composition_4"
    input_dict = {"operators": operators, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint,
                  "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    op1 = tf.linalg.LinearOperatorIdentity(num_rows=3)
    op2 = tf.linalg.LinearOperatorDiag(np.array([1., 2., 3.]))
    operators = [op1, op2]
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "test_composition_5"
    input_dict = {"operators": operators, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint,
                  "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    op1 = tf.linalg.LinearOperatorFullMatrix(np.array([[-1., 2.], [3., -4.]]))
    operators = [op1]
    is_non_singular = True
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "test_composition_6"
    input_dict = {"operators": operators, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint,
                  "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    matrix_34 = np.random.normal(size=[3, 4])
    op_34 = tf.linalg.LinearOperatorFullMatrix(matrix_34)
    matrix_42 = np.random.normal(size=[4, 2])
    op_42 = tf.linalg.LinearOperatorFullMatrix(matrix_42)
    operators = [op_34, op_42]
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = False
    name = "test_composition_7"
    input_dict = {"operators": operators, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint,
                  "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Batch of matrices
    matrix_2345 = np.random.normal(size=[2, 3, 4, 5])
    op_2345 = tf.linalg.LinearOperatorFullMatrix(matrix_2345)
    matrix_2356 = np.random.normal(size=[2, 3, 5, 6])
    op_2356 = tf.linalg.LinearOperatorFullMatrix(matrix_2356)
    operators = [op_2345, op_2356]
    is_non_singular = False
    is_self_adjoint = False
    is_positive_definite = False
    is_square = False
    name = "test_composition_8"
    input_dict = {"operators": operators, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint,
                  "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Composition of identity operators
    op1 = tf.linalg.LinearOperatorIdentity(num_rows=5)
    op2 = tf.linalg.LinearOperatorIdentity(num_rows=5)
    operators = [op1, op2]
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "test_composition_9"
    input_dict = {"operators": operators, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint,
                  "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    matrix_12 = np.random.normal(size=[1, 2])
    op_12 = tf.linalg.LinearOperatorFullMatrix(matrix_12)
    matrix_23 = np.random.normal(size=[2, 3])
    op_23 = tf.linalg.LinearOperatorFullMatrix(matrix_23)
    operators = [op_12, op_23]
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = False
    name = "test_composition_10"
    input_dict = {"operators": operators, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint,
                  "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.LinearOperatorComposition"] = tf_linalg_linearoperatorcomposition_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.LinearOperatorComposition' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorComposition'.")

check_valid('tf.linalg.LinearOperatorComposition', generated_inputs['tf.linalg.LinearOperatorComposition'], lib="tf", suffix=0)
