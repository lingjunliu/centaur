
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_tridiagonal_solve_inputs():
    list_of_inputs = []

    # Input 1: compact format, float32
    diagonals = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]], dtype=np.float32)
    rhs = np.array([[10.0, 11.0, 12.0]], dtype=np.float32)
    input_dict = {'diagonals': diagonals, 'rhs': rhs, 'diagonals_format': 'compact', 'transpose_rhs': False, 'conjugate_rhs': False, 'name': None, 'partial_pivoting': True, 'perturb_singular': False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: sequence format, float64 - fixed
    superdiagonal = np.array([[1.0, 2.0]], dtype=np.float64)
    main_diagonal = np.array([[3.0, 4.0, 5.0]], dtype=np.float64)
    subdiagonal = np.array([[6.0, 7.0]], dtype=np.float64)
    rhs = np.array([[8.0, 9.0, 10.0]], dtype=np.float64)
    input_dict = {'diagonals': (superdiagonal, main_diagonal, subdiagonal), 'rhs': rhs, 'diagonals_format': 'sequence', 'transpose_rhs': False, 'conjugate_rhs': False, 'name': None, 'partial_pivoting': True, 'perturb_singular': False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: matrix format, complex64
    diagonals = np.array([[[1+1j, 2, 0], [3, 4+1j, 5], [0, 6, 7+1j]]], dtype=np.complex64)
    rhs = np.array([[8+1j, 9, 10]], dtype=np.complex64)
    input_dict = {'diagonals': diagonals, 'rhs': rhs, 'diagonals_format': 'matrix', 'transpose_rhs': False, 'conjugate_rhs': False, 'name': None, 'partial_pivoting': True, 'perturb_singular': False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: compact format, complex128, multiple RHS
    diagonals = np.array([[[1+1j, 2, 3], [4, 5+1j, 6], [7, 8, 9+1j]]], dtype=np.complex128)
    rhs = np.array([[[10+1j, 11+1j], [12+1j, 13+1j], [14+1j, 15+1j]]], dtype=np.complex128)
    input_dict = {'diagonals': diagonals, 'rhs': rhs, 'diagonals_format': 'compact', 'transpose_rhs': False, 'conjugate_rhs': False, 'name': None, 'partial_pivoting': True, 'perturb_singular': False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: sequence format, float32, transpose_rhs=True - fixed
    superdiagonal = np.array([[1.0, 2.0]], dtype=np.float32)
    main_diagonal = np.array([[3.0, 4.0, 5.0]], dtype=np.float32)
    subdiagonal = np.array([[6.0, 7.0]], dtype=np.float32)
    rhs = np.array([[8.0, 9.0, 10.0]], dtype=np.float32)
    input_dict = {'diagonals': (superdiagonal, main_diagonal, subdiagonal), 'rhs': rhs, 'diagonals_format': 'sequence', 'transpose_rhs': True, 'conjugate_rhs': False, 'name': None, 'partial_pivoting': True, 'perturb_singular': False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: matrix format, float64, perturb_singular=True
    diagonals = np.array([[[0.0, 2, 0], [3, 0.0, 5], [0, 6, 0.0]]], dtype=np.float64)
    rhs = np.array([[8.0, 9, 10]], dtype=np.float64)
    input_dict = {'diagonals': diagonals, 'rhs': rhs, 'diagonals_format': 'matrix', 'transpose_rhs': False, 'conjugate_rhs': False, 'name': None, 'partial_pivoting': True, 'perturb_singular': True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: compact format, float32, multiple batches
    diagonals = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                          [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]], dtype=np.float32)
    rhs = np.array([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]], dtype=np.float32)
    input_dict = {'diagonals': diagonals, 'rhs': rhs, 'diagonals_format': 'compact', 'transpose_rhs': False, 'conjugate_rhs': False, 'name': None, 'partial_pivoting': True, 'perturb_singular': False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: sequence format, complex128, conjugate_rhs=True -fixed
    superdiagonal = np.array([[1+1j, 2+2j]], dtype=np.complex128)
    main_diagonal = np.array([[3+3j, 4+4j, 5+5j]], dtype=np.complex128)
    subdiagonal = np.array([[6+6j, 7+7j]], dtype=np.complex128)
    rhs = np.array([[8+8j, 9+9j, 10+10j]], dtype=np.complex128)
    input_dict = {'diagonals': (superdiagonal, main_diagonal, subdiagonal), 'rhs': rhs, 'diagonals_format': 'sequence', 'transpose_rhs': False, 'conjugate_rhs': True, 'name': None, 'partial_pivoting': True, 'perturb_singular': False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: matrix format, float32, no pivoting
    diagonals = np.array([[[1.0, 2, 0], [3, 4.0, 5], [0, 6, 7.0]]], dtype=np.float32)
    rhs = np.array([[8.0, 9, 10]], dtype=np.float32)
    input_dict = {'diagonals': diagonals, 'rhs': rhs, 'diagonals_format': 'matrix', 'transpose_rhs': False, 'conjugate_rhs': False, 'name': None, 'partial_pivoting': False, 'perturb_singular': False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: compact format, float64, different rhs shape
    diagonals = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], dtype=np.float64)
    rhs = np.array([[10.0, 11.0]], dtype=np.float64)
    input_dict = {'diagonals': diagonals, 'rhs': rhs, 'diagonals_format': 'compact', 'transpose_rhs': False, 'conjugate_rhs': False, 'name': None, 'partial_pivoting': True, 'perturb_singular': False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: sequence format, complex64, name - fixed
    superdiagonal = np.array([[1+1j]], dtype=np.complex64)
    main_diagonal = np.array([[3+3j]], dtype=np.complex64)
    subdiagonal = np.array([[6+6j]], dtype=np.complex64)
    rhs = np.array([[8+8j]], dtype=np.complex64)
    input_dict = {'diagonals': (superdiagonal, main_diagonal, subdiagonal), 'rhs': rhs, 'diagonals_format': 'sequence', 'transpose_rhs': False, 'conjugate_rhs': False, 'name': "solve_tridiagonal", 'partial_pivoting': True, 'perturb_singular': False}
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
