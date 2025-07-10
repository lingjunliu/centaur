
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_linear_operator_toeplitz_inputs():
    list_of_inputs = []

    # Input 1
    col = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32).numpy()
    row = tf.constant([1.0, 4.0, 5.0], dtype=tf.float32).numpy()
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = None
    name = "toeplitz_1"

    input_dict = {
        "col": col,
        "row": row,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    col = tf.constant([1.0, 0.0, -1.0, -2.0], dtype=tf.float64).numpy()
    row = tf.constant([1.0, 0.0, -1.0, -2.0], dtype=tf.float64).numpy()
    is_non_singular = True
    is_self_adjoint = False
    is_positive_definite = None
    is_square = True
    name = "toeplitz_2"

    input_dict = {
        "col": col,
        "row": row,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    col = tf.constant([1 + 1j, 2 + 2j, 3 + 3j], dtype=tf.complex64).numpy()
    row = tf.constant([1 + 1j, 4 + 4j, -5 - 5j], dtype=tf.complex64).numpy()
    is_non_singular = False
    is_self_adjoint = True
    is_positive_definite = None
    is_square = None
    name = "toeplitz_3"

    input_dict = {
        "col": col,
        "row": row,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Batch of Toeplitz matrices
    col = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32).numpy()
    row = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32).numpy()
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = None
    name = "toeplitz_4"

    input_dict = {
        "col": col,
        "row": row,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex numbers
    col = tf.constant([1j, 2j, 3j], dtype=tf.complex128).numpy()
    row = tf.constant([1j, 2j, 3j], dtype=tf.complex128).numpy()
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = None
    name = "toeplitz_5"

    input_dict = {
        "col": col,
        "row": row,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different shapes, same length
    col = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32).numpy()
    row = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32).numpy()
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = None
    name = "toeplitz_6"

    input_dict = {
        "col": col,
        "row": row,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7
    col = tf.constant([-1.0, -2.0, -3.0], dtype=tf.float32).numpy()
    row = tf.constant([-1.0, -2.0, -3.0], dtype=tf.float32).numpy()
    is_non_singular = True
    is_self_adjoint = False
    is_positive_definite = None
    is_square = True
    name = "toeplitz_7"

    input_dict = {
        "col": col,
        "row": row,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    col = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32).numpy()
    row = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32).numpy()
    is_non_singular = False
    is_self_adjoint = True
    is_positive_definite = None
    is_square = True
    name = "toeplitz_8"

    input_dict = {
        "col": col,
        "row": row,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    col = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32).numpy()
    row = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32).numpy()
    is_non_singular = True
    is_self_adjoint = None
    is_positive_definite = True
    is_square = None
    name = "toeplitz_9"

    input_dict = {
        "col": col,
        "row": row,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    col = tf.constant([1.0], dtype=tf.float32).numpy()
    row = tf.constant([1.0], dtype=tf.float32).numpy()
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = None
    name = "toeplitz_10"

    input_dict = {
        "col": col,
        "row": row,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.LinearOperatorToeplitz"] = tf_linalg_linear_operator_toeplitz_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.LinearOperatorToeplitz' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorToeplitz'.")

check_valid('tf.linalg.LinearOperatorToeplitz', generated_inputs['tf.linalg.LinearOperatorToeplitz'], lib="tf", suffix=0)
