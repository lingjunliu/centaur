
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_linearoperatorpermutation_inputs():
    list_of_inputs = []

    # Input 1
    perm = np.array([0, 1, 2], dtype=np.int32)
    dtype = tf.float32
    is_non_singular = True
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "perm1"
    input_dict = {"perm": perm, "dtype": dtype, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    perm = np.array([2, 0, 1], dtype=np.int32)
    dtype = tf.float64
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "perm2"
    input_dict = {"perm": perm, "dtype": dtype, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    perm = np.array([1, 0], dtype=np.int32)
    dtype = tf.float32
    is_non_singular = True
    is_self_adjoint = None
    is_positive_definite = None
    is_square = None
    name = "perm3"
    input_dict = {"perm": perm, "dtype": dtype, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    perm = np.array([[0, 1], [1, 0]], dtype=np.int32)
    dtype = tf.float32
    is_non_singular = True
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "perm4"
    input_dict = {"perm": perm, "dtype": dtype, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    perm = np.array([[0, 1, 2], [2, 0, 1]], dtype=np.int32)
    dtype = tf.float64
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "perm5"
    input_dict = {"perm": perm, "dtype": dtype, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    perm = np.array([0, 1, 2, 3], dtype=np.int32)
    dtype = tf.float32
    is_non_singular = True
    is_self_adjoint = None
    is_positive_definite = None
    is_square = None
    name = "perm6"
    input_dict = {"perm": perm, "dtype": dtype, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    perm = np.array([3, 2, 1, 0], dtype=np.int32)
    dtype = tf.float64
    is_non_singular = True
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "perm7"
    input_dict = {"perm": perm, "dtype": dtype, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8
    perm = np.array([[0, 2, 1], [1, 0, 2], [2, 1, 0]], dtype=np.int32)
    dtype = tf.float32
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "perm8"
    input_dict = {"perm": perm, "dtype": dtype, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    perm = np.array([0, 1], dtype=np.int32)
    dtype = tf.float32
    is_non_singular = True
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "perm9"
    input_dict = {"perm": perm, "dtype": dtype, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    perm = np.array([[1, 0], [0, 1]], dtype=np.int32)
    dtype = tf.float64
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "perm10"
    input_dict = {"perm": perm, "dtype": dtype, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.LinearOperatorPermutation"] = tf_linalg_linearoperatorpermutation_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.LinearOperatorPermutation' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorPermutation'.")

check_valid('tf.linalg.LinearOperatorPermutation', generated_inputs['tf.linalg.LinearOperatorPermutation'], lib="tf", suffix=0)
