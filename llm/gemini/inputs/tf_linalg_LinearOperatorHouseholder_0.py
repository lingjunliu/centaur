
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_linearoperatorhouseholder_inputs():
    list_of_inputs = []

    # Input 1
    reflection_axis = np.array([1.0, 0.0], dtype=np.float32)
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "householder_1"
    input_dict = {"reflection_axis": reflection_axis, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    reflection_axis = np.array([0.0, 1.0], dtype=np.float32)
    is_non_singular = False
    is_self_adjoint = True
    is_positive_definite = None
    is_square = True
    name = "householder_2"
    input_dict = {"reflection_axis": reflection_axis, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    reflection_axis = np.array([1.0, 1.0], dtype=np.float32)
    is_non_singular = None
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "householder_3"
    input_dict = {"reflection_axis": reflection_axis, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Batch of reflection axes
    reflection_axis = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "householder_4"
    input_dict = {"reflection_axis": reflection_axis, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger reflection axis
    reflection_axis = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    is_non_singular = False
    is_self_adjoint = True
    is_positive_definite = None
    is_square = True
    name = "householder_5"
    input_dict = {"reflection_axis": reflection_axis, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6: Batch and larger
    reflection_axis = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]], dtype=np.float32)
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "householder_6"
    input_dict = {"reflection_axis": reflection_axis, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different name
    reflection_axis = np.array([0.5, 0.5], dtype=np.float32)
    is_non_singular = None
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "another_householder"
    input_dict = {"reflection_axis": reflection_axis, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative values in axis
    reflection_axis = np.array([-1.0, 1.0], dtype=np.float32)
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "householder_8"
    input_dict = {"reflection_axis": reflection_axis, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    reflection_axis = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    is_non_singular = False
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "householder_9"
    input_dict = {"reflection_axis": reflection_axis, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    reflection_axis = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "householder_10"
    input_dict = {"reflection_axis": reflection_axis, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.LinearOperatorHouseholder"] = tf_linalg_linearoperatorhouseholder_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.LinearOperatorHouseholder' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorHouseholder'.")

check_valid('tf.linalg.LinearOperatorHouseholder', generated_inputs['tf.linalg.LinearOperatorHouseholder'], lib="tf", suffix=0)
