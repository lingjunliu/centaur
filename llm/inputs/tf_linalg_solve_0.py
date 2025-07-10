
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_solve_inputs():
    list_of_inputs = []

    # Input 1: Basic case with float32
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    rhs = np.array([[5.0], [6.0]], dtype=np.float32)
    adjoint = False
    name = "solve_1"
    input_dict = {"matrix": matrix, "rhs": rhs, "adjoint": adjoint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Complex64, adjoint=True
    matrix = np.array([[1.0 + 1j, 2.0], [3.0, 4.0 - 1j]], dtype=np.complex64)
    rhs = np.array([[5.0 + 1j], [6.0 - 1j]], dtype=np.complex64)
    adjoint = True
    name = "solve_2"
    input_dict = {"matrix": matrix, "rhs": rhs, "adjoint": adjoint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64, multi-dimensional
    matrix = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64)
    rhs = np.array([[[5.0], [6.0]], [[7.0], [8.0]]], dtype=np.float64)
    adjoint = False
    name = "solve_3"
    input_dict = {"matrix": matrix, "rhs": rhs, "adjoint": adjoint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: complex128, adjoint=True, different rhs dimension
    matrix = np.array([[1.0 + 1j, 2.0], [3.0, 4.0 - 1j]], dtype=np.complex128)
    rhs = np.array([[5.0 + 1j, 1.0], [6.0 - 1j, 2.0]], dtype=np.complex128)
    adjoint = True
    name = "solve_5"
    input_dict = {"matrix": matrix, "rhs": rhs, "adjoint": adjoint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: adjoint = False
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    rhs = np.array([[5.0, 7.0], [6.0, 8.0]], dtype=np.float32)
    adjoint = False
    name = "solve_6"
    input_dict = {"matrix": matrix, "rhs": rhs, "adjoint": adjoint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float32, identity matrix
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    rhs = np.array([[5.0], [6.0]], dtype=np.float32)
    adjoint = False
    name = "solve_7"
    input_dict = {"matrix": matrix, "rhs": rhs, "adjoint": adjoint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32, matrix with negative values
    matrix = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    rhs = np.array([[5.0], [-6.0]], dtype=np.float32)
    adjoint = False
    name = "solve_8"
    input_dict = {"matrix": matrix, "rhs": rhs, "adjoint": adjoint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float32, larger rhs
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    rhs = np.array([[5.0, 6.0, 7.0], [6.0, 7.0, 8.0]], dtype=np.float32)
    adjoint = False
    name = "solve_10"
    input_dict = {"matrix": matrix, "rhs": rhs, "adjoint": adjoint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float64, 3D matrix and RHS
    matrix = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64)
    rhs = np.array([[[5.0], [6.0]], [[7.0], [8.0]]], dtype=np.float64)
    adjoint = True
    name = "solve_11"
    input_dict = {"matrix": matrix, "rhs": rhs, "adjoint": adjoint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.solve"] = tf_linalg_solve_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.solve' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.solve'.")

check_valid('tf.linalg.solve', generated_inputs['tf.linalg.solve'], lib="tf", suffix=0)
