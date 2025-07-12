
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_triangular_solve_inputs():
    list_of_inputs = []

    # Input 1
    matrix = np.array([[3, 0, 0], [2, 1, 0], [1, 0, 1]], dtype=np.float32)
    rhs = np.array([[4], [2], [4]], dtype=np.float32)
    lower = True
    adjoint = False
    name = "solve1"
    input_dict = {"matrix": matrix, "rhs": rhs, "lower": lower, "adjoint": adjoint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    matrix = np.array([[1, 2, 3], [0, 4, 5], [0, 0, 6]], dtype=np.float32)
    rhs = np.array([[7], [8], [9]], dtype=np.float32)
    lower = False
    adjoint = False
    name = "solve2"
    input_dict = {"matrix": matrix, "rhs": rhs, "lower": lower, "adjoint": adjoint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    matrix = np.array([[1, 0], [2, 3]], dtype=np.float64)
    rhs = np.array([[4, 5], [6, 7]], dtype=np.float64)
    lower = True
    adjoint = True
    name = "solve3"
    input_dict = {"matrix": matrix, "rhs": rhs, "lower": lower, "adjoint": adjoint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    matrix = np.array([[4, 5], [0, 6]], dtype=np.float64)
    rhs = np.array([[7, 8], [9, 10]], dtype=np.float64)
    lower = False
    adjoint = True
    name = "solve4"
    input_dict = {"matrix": matrix, "rhs": rhs, "lower": lower, "adjoint": adjoint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    matrix = np.array([[[1, 0], [2, 3]], [[4, 0], [5, 6]]], dtype=np.float32)
    rhs = np.array([[[4, 5], [6, 7]], [[7, 8], [9, 10]]], dtype=np.float32)
    lower = True
    adjoint = False
    name = "solve5"
    input_dict = {"matrix": matrix, "rhs": rhs, "lower": lower, "adjoint": adjoint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    matrix = np.array([[[1, 2], [0, 3]], [[4, 5], [0, 6]]], dtype=np.float32)
    rhs = np.array([[[4, 5], [6, 7]], [[7, 8], [9, 10]]], dtype=np.float32)
    lower = False
    adjoint = False
    name = "solve6"
    input_dict = {"matrix": matrix, "rhs": rhs, "lower": lower, "adjoint": adjoint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7
    matrix = np.array([[3, 0, 0], [2, 1, 0], [1, 0, 1]], dtype=np.complex64)
    rhs = np.array([[4+1j], [2+2j], [4+3j]], dtype=np.complex64)
    lower = True
    adjoint = False
    name = "solve7"
    input_dict = {"matrix": matrix, "rhs": rhs, "lower": lower, "adjoint": adjoint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    matrix = np.array([[1, 2, 3], [0, 4, 5], [0, 0, 6]], dtype=np.complex128)
    rhs = np.array([[7+1j], [8+2j], [9+3j]], dtype=np.complex128)
    lower = False
    adjoint = True
    name = "solve8"
    input_dict = {"matrix": matrix, "rhs": rhs, "lower": lower, "adjoint": adjoint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    matrix = np.array([[1, 0], [2, 3]], dtype=np.float32)
    rhs = np.array([[4, 5, 6], [6, 7, 8]], dtype=np.float32)
    lower = True
    adjoint = False
    name = "solve9"
    input_dict = {"matrix": matrix, "rhs": rhs, "lower": lower, "adjoint": adjoint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    matrix = np.array([[4, 5], [0, 6]], dtype=np.float32)
    rhs = np.array([[7, 8, 9], [9, 10, 11]], dtype=np.float32)
    lower = False
    adjoint = True
    name = "solve10"
    input_dict = {"matrix": matrix, "rhs": rhs, "lower": lower, "adjoint": adjoint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.triangular_solve"] = tf_linalg_triangular_solve_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.triangular_solve' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.triangular_solve'.")

check_valid('tf.linalg.triangular_solve', generated_inputs['tf.linalg.triangular_solve'], lib="tf", suffix=0)
