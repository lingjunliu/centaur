
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_tridiagonal_solve_inputs():
    list_of_inputs = []

    # Input 1: compact format, single RHS
    diagonals = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=np.float32)
    rhs = np.array([[10, 11, 12]], dtype=np.float32)
    input_dict = {"diagonals": diagonals, "rhs": rhs, "diagonals_format": "compact", "transpose_rhs": False, "conjugate_rhs": False, "name": None, "partial_pivoting": True, "perturb_singular": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: compact format, multiple RHS (batch size matching)
    diagonals = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[11, 12, 13], [14, 15, 16], [17, 18, 19]]], dtype=np.float32)
    rhs = np.array([[10, 11, 12], [13, 14, 15]], dtype=np.float32)
    input_dict = {"diagonals": diagonals, "rhs": rhs, "diagonals_format": "compact", "transpose_rhs": False, "conjugate_rhs": False, "name": None, "partial_pivoting": True, "perturb_singular": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: sequence format, single RHS
    superdiag = np.array([[1, 2]], dtype=np.float32)
    maindiag = np.array([[3, 4, 5]], dtype=np.float32)
    subdiag = np.array([[6, 7]], dtype=np.float32)
    rhs = np.array([[8, 9, 10]], dtype=np.float32)
    input_dict = {"diagonals": (superdiag, maindiag, subdiag), "rhs": rhs, "diagonals_format": "sequence", "transpose_rhs": False, "conjugate_rhs": False, "name": None, "partial_pivoting": True, "perturb_singular": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: sequence format, multiple RHS (batch size matching)
    superdiag = np.array([[1, 2], [11, 12]], dtype=np.float32)
    maindiag = np.array([[3, 4, 5], [13, 14, 15]], dtype=np.float32)
    subdiag = np.array([[6, 7], [16, 17]], dtype=np.float32)
    rhs = np.array([[8, 9, 10], [11, 12, 13]], dtype=np.float32)
    input_dict = {"diagonals": (superdiag, maindiag, subdiag), "rhs": rhs, "diagonals_format": "sequence", "transpose_rhs": False, "conjugate_rhs": False, "name": None, "partial_pivoting": True, "perturb_singular": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: matrix format, single RHS
    diagonals = np.array([[[1, 2, 0], [4, 5, 6], [0, 8, 9]]], dtype=np.float32)
    rhs = np.array([[10, 11, 12]], dtype=np.float32)
    input_dict = {"diagonals": diagonals, "rhs": rhs, "diagonals_format": "matrix", "transpose_rhs": False, "conjugate_rhs": False, "name": None, "partial_pivoting": True, "perturb_singular": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: matrix format, multiple RHS (batch size matching)
    diagonals = np.array([[[1, 2, 0], [4, 5, 6], [0, 8, 9]], [[11, 12, 0], [14, 15, 16], [0, 18, 19]]], dtype=np.float32)
    rhs = np.array([[10, 11, 12], [13, 14, 15]], dtype=np.float32)
    input_dict = {"diagonals": diagonals, "rhs": rhs, "diagonals_format": "matrix", "transpose_rhs": False, "conjugate_rhs": False, "name": None, "partial_pivoting": True, "perturb_singular": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: compact format, complex numbers
    diagonals = np.array([[[1+1j, 2+2j, 3+3j], [4+4j, 5+5j, 6+6j], [7+7j, 8+8j, 9+9j]]], dtype=np.complex64)
    rhs = np.array([[10+10j, 11+11j, 12+12j]], dtype=np.complex64)
    input_dict = {"diagonals": diagonals, "rhs": rhs, "diagonals_format": "compact", "transpose_rhs": False, "conjugate_rhs": False, "name": None, "partial_pivoting": True, "perturb_singular": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8: compact format, with transpose_rhs
    diagonals = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=np.float32)
    rhs = np.array([[10, 11, 12]], dtype=np.float32)
    input_dict = {"diagonals": diagonals, "rhs": rhs, "diagonals_format": "compact", "transpose_rhs": True, "conjugate_rhs": False, "name": None, "partial_pivoting": True, "perturb_singular": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: compact format, with conjugate_rhs
    diagonals = np.array([[[1+1j, 2+2j, 3+3j], [4+4j, 5+5j, 6+6j], [7+7j, 8+8j, 9+9j]]], dtype=np.complex64)
    rhs = np.array([[10+10j, 11+11j, 12+12j]], dtype=np.complex64)
    input_dict = {"diagonals": diagonals, "rhs": rhs, "diagonals_format": "compact", "transpose_rhs": False, "conjugate_rhs": True, "name": None, "partial_pivoting": True, "perturb_singular": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: compact format, with perturb_singular
    diagonals = np.array([[[1e-8, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=np.float32)
    rhs = np.array([[10, 11, 12]], dtype=np.float32)
    input_dict = {"diagonals": diagonals, "rhs": rhs, "diagonals_format": "compact", "transpose_rhs": False, "conjugate_rhs": False, "name": None, "partial_pivoting": True, "perturb_singular": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: compact format, 2D, multiple RHS, perturb_singular = True
    diagonals = np.array([[[1e-08, 2.], [3., 4.], [5., 6.]], [[7., 8.], [9., 10.], [11., 12.]]], dtype=np.float32)
    rhs = np.array([[10, 11], [12,13]], dtype=np.float32)
    input_dict = {"diagonals": diagonals, "rhs": rhs, "diagonals_format": "compact", "transpose_rhs": False, "conjugate_rhs": False, "name": None, "partial_pivoting": True, "perturb_singular": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: sequence format, different M and N
    superdiag = np.array([[1, 2, 3]], dtype=np.float32)
    maindiag = np.array([[4, 5, 6, 7]], dtype=np.float32)
    subdiag = np.array([[8, 9, 10]], dtype=np.float32)
    rhs = np.array([[11, 12, 13, 14]], dtype=np.float32)
    input_dict = {"diagonals": (superdiag, maindiag, subdiag), "rhs": rhs, "diagonals_format": "sequence", "transpose_rhs": False, "conjugate_rhs": False, "name": None, "partial_pivoting": True, "perturb_singular": False}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 13: sequence format, different M and N, multiple RHS
    superdiag = np.array([[1, 2, 3], [15, 16, 17]], dtype=np.float32)
    maindiag = np.array([[4, 5, 6, 7], [18, 19, 20, 21]], dtype=np.float32)
    subdiag = np.array([[8, 9, 10], [22, 23, 24]], dtype=np.float32)
    rhs = np.array([[11, 12, 13, 14], [25, 26, 27, 28]], dtype=np.float32)
    input_dict = {"diagonals": (superdiag, maindiag, subdiag), "rhs": rhs, "diagonals_format": "sequence", "transpose_rhs": False, "conjugate_rhs": False, "name": None, "partial_pivoting": True, "perturb_singular": False}
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
