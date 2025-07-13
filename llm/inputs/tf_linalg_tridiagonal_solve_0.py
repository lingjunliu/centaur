
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_tridiagonal_solve_inputs():
    list_of_inputs = []

    # Input 1: compact format, basic case
    diagonals = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=np.float32)
    rhs = np.array([[10, 11, 12]], dtype=np.float32)
    input_dict = {"diagonals": diagonals, "rhs": rhs, "diagonals_format": 'compact', "transpose_rhs": False, "conjugate_rhs": False, "name": None, "partial_pivoting": True, "perturb_singular": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: compact format, multiple rhs
    diagonals = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=np.float32)
    rhs = np.array([[10, 13], [11, 14], [12, 15]], dtype=np.float32)
    input_dict = {"diagonals": diagonals, "rhs": rhs, "diagonals_format": 'compact', "transpose_rhs": False, "conjugate_rhs": False, "name": None, "partial_pivoting": True, "perturb_singular": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: sequence format
    superdiag = np.array([[1, 2]], dtype=np.float32)
    diag = np.array([[4, 5, 6]], dtype=np.float32)
    subdiag = np.array([[7, 8]], dtype=np.float32)
    diagonals = (superdiag, diag, subdiag)
    rhs = np.array([[10, 11, 12]], dtype=np.float32)
    input_dict = {"diagonals": diagonals, "rhs": rhs, "diagonals_format": 'sequence', "transpose_rhs": False, "conjugate_rhs": False, "name": None, "partial_pivoting": True, "perturb_singular": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: matrix format
    diagonals = np.array([[[1, 2, 0], [7, 4, 5], [0, 8, 6]]], dtype=np.float32)
    rhs = np.array([[10, 11, 12]], dtype=np.float32)
    input_dict = {"diagonals": diagonals, "rhs": rhs, "diagonals_format": 'matrix', "transpose_rhs": False, "conjugate_rhs": False, "name": None, "partial_pivoting": True, "perturb_singular": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex numbers
    diagonals = np.array([[[1+1j, 2+2j, 3+3j], [4+4j, 5+5j, 6+6j], [7+7j, 8+8j, 9+9j]]], dtype=np.complex64)
    rhs = np.array([[10+10j, 11+11j, 12+12j]], dtype=np.complex64)
    input_dict = {"diagonals": diagonals, "rhs": rhs, "diagonals_format": 'compact', "transpose_rhs": False, "conjugate_rhs": False, "name": None, "partial_pivoting": True, "perturb_singular": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: transpose_rhs=True - Corrected
    diagonals = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=np.float32)
    rhs = np.array([[10], [11], [12]], dtype=np.float32)  # Reshape rhs
    input_dict = {"diagonals": diagonals, "rhs": rhs, "diagonals_format": 'compact', "transpose_rhs": True, "conjugate_rhs": False, "name": None, "partial_pivoting": True, "perturb_singular": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: conjugate_rhs=True, complex
    diagonals = np.array([[[1+1j, 2+2j, 3+3j], [4+4j, 5+5j, 6+6j], [7+7j, 8+8j, 9+9j]]], dtype=np.complex64)
    rhs = np.array([[10+10j, 11+11j, 12+12j]], dtype=np.complex64)
    input_dict = {"diagonals": diagonals, "rhs": rhs, "diagonals_format": 'compact', "transpose_rhs": False, "conjugate_rhs": True, "name": None, "partial_pivoting": True, "perturb_singular": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: partial_pivoting=False
    diagonals = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=np.float32)
    rhs = np.array([[10, 11, 12]], dtype=np.float32)
    input_dict = {"diagonals": diagonals, "rhs": rhs, "diagonals_format": 'compact', "transpose_rhs": False, "conjugate_rhs": False, "name": None, "partial_pivoting": False, "perturb_singular": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: perturb_singular=True
    diagonals = np.array([[[1e-8, 2, 3], [4, 1e-8, 6], [7, 8, 1e-8]]], dtype=np.float32)
    rhs = np.array([[10, 11, 12]], dtype=np.float32)
    input_dict = {"diagonals": diagonals, "rhs": rhs, "diagonals_format": 'compact', "transpose_rhs": False, "conjugate_rhs": False, "name": None, "partial_pivoting": True, "perturb_singular": True}
    list_of_inputs.append(copy.deepcopy(input_dict))
  
    # Input 10: batch dimension
    diagonals = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[9, 8, 7], [6, 5, 4], [3, 2, 1]]], dtype=np.float32)
    rhs = np.array([[10, 11, 12], [13, 14, 15]], dtype=np.float32)
    input_dict = {"diagonals": diagonals, "rhs": rhs, "diagonals_format": 'compact', "transpose_rhs": False, "conjugate_rhs": False, "name": None, "partial_pivoting": True, "perturb_singular": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.tridiagonal_solve"] = tf_linalg_tridiagonal_solve_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.tridiagonal_solve' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.tridiagonal_solve'.")

check_valid('tf.linalg.tridiagonal_solve', generated_inputs['tf.linalg.tridiagonal_solve'], lib="tf", suffix=0)
