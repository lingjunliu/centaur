
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_linear_operator_diag_inputs():
    list_of_inputs = []

    # Input 1
    diag = tf.constant([1.0, 2.0, 3.0]).numpy()
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "diag_op_1"
    input_dict = {"diag": diag, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint,
                  "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    diag = tf.constant([-1.0, 2.0, -3.0]).numpy()
    is_non_singular = None
    is_self_adjoint = True
    is_positive_definite = None
    is_square = None
    name = "diag_op_2"
    input_dict = {"diag": diag, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint,
                  "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    diag = tf.constant([[1.0, 2.0], [3.0, 4.0]]).numpy()
    is_non_singular = False
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "diag_op_3"
    input_dict = {"diag": diag, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint,
                  "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    diag = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).numpy()
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = False
    is_square = None
    name = "diag_op_4"
    input_dict = {"diag": diag, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint,
                  "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    diag = tf.constant([1.0, 0.0, 3.0]).numpy()
    is_non_singular = False
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "diag_op_5"
    input_dict = {"diag": diag, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint,
                  "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    diag = tf.constant([[-1.0, 0.0], [0.0, -1.0]]).numpy()
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "diag_op_6"
    input_dict = {"diag": diag, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint,
                  "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    diag = tf.constant([1.5, 2.5, 3.5]).numpy()
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "diag_op_7"
    input_dict = {"diag": diag, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint,
                  "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    diag = tf.constant([[-1.5, 0.0], [0.0, -2.5]]).numpy()
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "diag_op_8"
    input_dict = {"diag": diag, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint,
                  "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    diag = tf.constant([1, 2, 3], dtype=tf.float32).numpy()
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "diag_op_9"
    input_dict = {"diag": diag, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint,
                  "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    diag = tf.constant([1.0], dtype=tf.float32).numpy()
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "diag_op_10"
    input_dict = {"diag": diag, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint,
                  "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    diag = tf.constant([1 + 1j, 2 + 2j]).numpy()
    is_non_singular = True
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "diag_op_11"
    input_dict = {"diag": diag, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint,
                  "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.LinearOperatorDiag"] = tf_linalg_linear_operator_diag_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.LinearOperatorDiag' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorDiag'.")

check_valid('tf.linalg.LinearOperatorDiag', generated_inputs['tf.linalg.LinearOperatorDiag'], lib="tf", suffix=0)
