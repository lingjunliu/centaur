
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_linearoperatorlowertriangular_inputs():
    list_of_inputs = []

    # Input 1
    tril = np.array([[1.0, 0.0], [2.0, 3.0]], dtype=np.float32)
    is_non_singular = True
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "lower_triangular_1"
    input_dict = {"tril": tril, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    tril = np.array([[[1.0, 0.0], [2.0, 3.0]], [[4.0, 0.0], [5.0, 6.0]]], dtype=np.float32)
    is_non_singular = True
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "lower_triangular_2"
    input_dict = {"tril": tril, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    tril = np.array([[1.0, 0.0, 0.0], [2.0, 3.0, 0.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = True
    name = "lower_triangular_3"
    input_dict = {"tril": tril, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    tril = np.random.rand(2, 3, 3).astype(np.float32)
    is_non_singular = True
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "lower_triangular_4"
    input_dict = {"tril": tril, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 5
    tril = np.array([[1.0, 0.0], [-2.0, -3.0]], dtype=np.float32)
    is_non_singular = True
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "lower_triangular_5"
    input_dict = {"tril": tril, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    tril = np.array([[[1.0, 0.0], [-2.0, -3.0]], [[-4.0, 0.0], [5.0, -6.0]]], dtype=np.float32)
    is_non_singular = True
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "lower_triangular_6"
    input_dict = {"tril": tril, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    tril = np.array([[1.0, 0.0, 0.0], [-2.0, -3.0, 0.0], [-4.0, -5.0, -6.0]], dtype=np.float32)
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = True
    name = "lower_triangular_7"
    input_dict = {"tril": tril, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    tril = np.random.rand(4, 2, 2).astype(np.float32)
    is_non_singular = True
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "lower_triangular_8"
    input_dict = {"tril": tril, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    tril = np.array([[5.0, 0.0], [0.0, 7.0]], dtype=np.float32)
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "lower_triangular_9"
    input_dict = {"tril": tril, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    tril = np.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "lower_triangular_10"
    input_dict = {"tril": tril, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.LinearOperatorLowerTriangular"] = tf_linalg_linearoperatorlowertriangular_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.LinearOperatorLowerTriangular' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorLowerTriangular'.")

check_valid('tf.linalg.LinearOperatorLowerTriangular', generated_inputs['tf.linalg.LinearOperatorLowerTriangular'], lib="tf", suffix=0)
