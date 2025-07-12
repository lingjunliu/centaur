
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_linearoperatoridentity_inputs():
    list_of_inputs = []

    # Input 1
    num_rows = 2
    batch_shape = None
    dtype = np.float32
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    assert_proper_shapes = False
    name = "identity_1"
    input_dict = {"num_rows": num_rows, "batch_shape": [], "dtype": dtype, "is_non_singular": is_non_singular, "is_self_adjoint": True, "is_positive_definite": is_positive_definite, "is_square": is_square, "assert_proper_shapes": assert_proper_shapes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    num_rows = 3
    batch_shape = [2]
    dtype = np.int32
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    assert_proper_shapes = True
    name = "identity_2"
    input_dict = {"num_rows": num_rows, "batch_shape": [2], "dtype": dtype, "is_non_singular": is_non_singular, "is_self_adjoint": True, "is_positive_definite": is_positive_definite, "is_square": is_square, "assert_proper_shapes": assert_proper_shapes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    num_rows = 4
    batch_shape = [2, 3]
    dtype = np.float64
    is_non_singular = False
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    assert_proper_shapes = False
    name = "identity_3"
    input_dict = {"num_rows": num_rows, "batch_shape": [2, 3], "dtype": dtype, "is_non_singular": is_non_singular, "is_self_adjoint": True, "is_positive_definite": is_positive_definite, "is_square": is_square, "assert_proper_shapes": assert_proper_shapes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    num_rows = 5
    batch_shape = [1, 2, 1]
    dtype = np.complex64
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = None
    is_square = True
    assert_proper_shapes = True
    name = "identity_4"
    input_dict = {"num_rows": num_rows, "batch_shape": [1, 2, 1], "dtype": dtype, "is_non_singular": is_non_singular, "is_self_adjoint": True, "is_positive_definite": is_positive_definite, "is_square": is_square, "assert_proper_shapes": assert_proper_shapes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    num_rows = 1
    batch_shape = []
    dtype = np.int64
    is_non_singular = None
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    assert_proper_shapes = False
    name = "identity_5"
    input_dict = {"num_rows": num_rows, "batch_shape": [], "dtype": dtype, "is_non_singular": is_non_singular, "is_self_adjoint": True, "is_positive_definite": is_positive_definite, "is_square": is_square, "assert_proper_shapes": assert_proper_shapes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6
    num_rows = 7
    batch_shape = [5]
    dtype = np.uint8
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    assert_proper_shapes = True
    name = "identity_6"
    input_dict = {"num_rows": num_rows, "batch_shape": [5], "dtype": dtype, "is_non_singular": is_non_singular, "is_self_adjoint": True, "is_positive_definite": is_positive_definite, "is_square": is_square, "assert_proper_shapes": assert_proper_shapes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    num_rows = 6
    batch_shape = [4, 2]
    dtype = np.bool_
    is_non_singular = False
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    assert_proper_shapes = False
    name = "identity_7"
    input_dict = {"num_rows": num_rows, "batch_shape": [4, 2], "dtype": dtype, "is_non_singular": is_non_singular, "is_self_adjoint": True, "is_positive_definite": is_positive_definite, "is_square": is_square, "assert_proper_shapes": assert_proper_shapes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    num_rows = 8
    batch_shape = [2, 2, 2]
    dtype = np.complex128
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = None
    is_square = True
    assert_proper_shapes = True
    name = "identity_8"
    input_dict = {"num_rows": num_rows, "batch_shape": [2, 2, 2], "dtype": dtype, "is_non_singular": is_non_singular, "is_self_adjoint": True, "is_positive_definite": is_positive_definite, "is_square": is_square, "assert_proper_shapes": assert_proper_shapes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    num_rows = 9
    batch_shape = [3]
    dtype = np.float16
    is_non_singular = None
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    assert_proper_shapes = False
    name = "identity_9"
    input_dict = {"num_rows": num_rows, "batch_shape": [3], "dtype": dtype, "is_non_singular": is_non_singular, "is_self_adjoint": True, "is_positive_definite": is_positive_definite, "is_square": is_square, "assert_proper_shapes": assert_proper_shapes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    num_rows = 10
    batch_shape = None
    dtype = np.int8
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    assert_proper_shapes = True
    name = "identity_10"
    input_dict = {"num_rows": num_rows, "batch_shape": [], "dtype": dtype, "is_non_singular": is_non_singular, "is_self_adjoint": True, "is_positive_definite": is_positive_definite, "is_square": is_square, "assert_proper_shapes": assert_proper_shapes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.LinearOperatorIdentity"] = tf_linalg_linearoperatoridentity_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.LinearOperatorIdentity' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorIdentity'.")

check_valid('tf.linalg.LinearOperatorIdentity', generated_inputs['tf.linalg.LinearOperatorIdentity'], lib="tf", suffix=0)
