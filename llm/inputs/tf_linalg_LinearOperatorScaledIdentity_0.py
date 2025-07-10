
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_linear_operator_scaled_identity_inputs():
    list_of_inputs = []

    # Input 1
    num_rows = 2
    multiplier = tf.constant(3.0).numpy()
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    assert_proper_shapes = False
    name = "ScaledIdentity1"
    input_dict = {"num_rows": num_rows, "multiplier": multiplier, "is_non_singular": is_non_singular,
                  "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite,
                  "is_square": is_square, "assert_proper_shapes": assert_proper_shapes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    num_rows = 3
    multiplier = tf.constant([1.0, 2.0]).numpy()
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    assert_proper_shapes = True
    name = "ScaledIdentity2"
    input_dict = {"num_rows": num_rows, "multiplier": multiplier, "is_non_singular": is_non_singular,
                  "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite,
                  "is_square": is_square, "assert_proper_shapes": assert_proper_shapes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    num_rows = 4
    multiplier = tf.constant([[1.0, 2.0], [3.0, 4.0]]).numpy()
    is_non_singular = None
    is_self_adjoint = True
    is_positive_definite = None
    is_square = True
    assert_proper_shapes = False
    name = "ScaledIdentity3"
    input_dict = {"num_rows": num_rows, "multiplier": multiplier, "is_non_singular": is_non_singular,
                  "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite,
                  "is_square": is_square, "assert_proper_shapes": assert_proper_shapes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    num_rows = 5
    multiplier = tf.constant(-2.0).numpy()
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    assert_proper_shapes = True
    name = "ScaledIdentity4"
    input_dict = {"num_rows": num_rows, "multiplier": multiplier, "is_non_singular": is_non_singular,
                  "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite,
                  "is_square": is_square, "assert_proper_shapes": assert_proper_shapes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 5
    num_rows = 1
    multiplier = tf.constant(0.0).numpy()
    is_non_singular = False
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    assert_proper_shapes = False
    name = "ScaledIdentity5"
    input_dict = {"num_rows": num_rows, "multiplier": multiplier, "is_non_singular": is_non_singular,
                  "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite,
                  "is_square": is_square, "assert_proper_shapes": assert_proper_shapes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    num_rows = 6
    multiplier = tf.constant([[-1.0, -2.0], [-3.0, -4.0]]).numpy()
    is_non_singular = None
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    assert_proper_shapes = True
    name = "ScaledIdentity6"
    input_dict = {"num_rows": num_rows, "multiplier": multiplier, "is_non_singular": is_non_singular,
                  "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite,
                  "is_square": is_square, "assert_proper_shapes": assert_proper_shapes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    num_rows = 7
    multiplier = tf.constant([1.5, 2.5, 3.5]).numpy()
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    assert_proper_shapes = False
    name = "ScaledIdentity7"
    input_dict = {"num_rows": num_rows, "multiplier": multiplier, "is_non_singular": is_non_singular,
                  "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite,
                  "is_square": is_square, "assert_proper_shapes": assert_proper_shapes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    num_rows = 8
    multiplier = tf.constant(4.0, dtype=tf.float64).numpy()
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    assert_proper_shapes = True
    name = "ScaledIdentity8"
    input_dict = {"num_rows": num_rows, "multiplier": multiplier, "is_non_singular": is_non_singular,
                  "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite,
                  "is_square": is_square, "assert_proper_shapes": assert_proper_shapes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    num_rows = 9
    multiplier = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).numpy()
    is_non_singular = None
    is_self_adjoint = True
    is_positive_definite = None
    is_square = True
    assert_proper_shapes = False
    name = "ScaledIdentity9"
    input_dict = {"num_rows": num_rows, "multiplier": multiplier, "is_non_singular": is_non_singular,
                  "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite,
                  "is_square": is_square, "assert_proper_shapes": assert_proper_shapes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    num_rows = 10
    multiplier = tf.constant(1.0).numpy()
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    assert_proper_shapes = False
    name = "ScaledIdentity10"
    input_dict = {"num_rows": num_rows, "multiplier": multiplier, "is_non_singular": is_non_singular,
                  "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite,
                  "is_square": is_square, "assert_proper_shapes": assert_proper_shapes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.LinearOperatorScaledIdentity"] = tf_linalg_linear_operator_scaled_identity_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.LinearOperatorScaledIdentity' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorScaledIdentity'.")

check_valid('tf.linalg.LinearOperatorScaledIdentity', generated_inputs['tf.linalg.LinearOperatorScaledIdentity'], lib="tf", suffix=0)
