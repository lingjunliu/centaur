
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_linearoperatorinversion_inputs():
    list_of_inputs = []

    # Input 1
    operator = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "inv_op_1"
    operator = tf.linalg.LinearOperatorFullMatrix(operator)
    input_dict = {"operator": operator, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    operator = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "inv_op_2"
    operator = tf.linalg.LinearOperatorFullMatrix(operator)
    input_dict = {"operator": operator, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    operator = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "inv_op_3"
    operator = tf.linalg.LinearOperatorFullMatrix(operator)

    input_dict = {"operator": operator, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    operator = np.array([[4.0, 0.0], [0.0, 9.0]], dtype=np.float32)
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "inv_op_4"
    operator = tf.linalg.LinearOperatorFullMatrix(operator)
    input_dict = {"operator": operator, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    operator = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float32)
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "inv_op_5"
    operator = tf.linalg.LinearOperatorFullMatrix(operator)
    input_dict = {"operator": operator, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    operator = np.array([[2.0, 0.0], [0.0, 0.5]], dtype=np.float32)
    is_non_singular = True
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "inv_op_6"
    operator = tf.linalg.LinearOperatorFullMatrix(operator)
    input_dict = {"operator": operator, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    operator = np.array([[5.0, 2.0], [2.0, 5.0]], dtype=np.float32)
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "inv_op_7"
    operator = tf.linalg.LinearOperatorFullMatrix(operator)
    input_dict = {"operator": operator, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    operator = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = None
    name = "inv_op_8"
    operator = tf.linalg.LinearOperatorFullMatrix(operator)
    input_dict = {"operator": operator, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9
    operator = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]], dtype=np.float32)
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "inv_op_9"
    operator = tf.linalg.LinearOperatorFullMatrix(operator)
    input_dict = {"operator": operator, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    operator = np.array([[0.5, 0.0], [0.0, 0.25]], dtype=np.float32)
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "inv_op_10"
    operator = tf.linalg.LinearOperatorFullMatrix(operator)
    input_dict = {"operator": operator, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.LinearOperatorInversion"] = tf_linalg_linearoperatorinversion_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.LinearOperatorInversion' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorInversion'.")

check_valid('tf.linalg.LinearOperatorInversion', generated_inputs['tf.linalg.LinearOperatorInversion'], lib="tf", suffix=0)
