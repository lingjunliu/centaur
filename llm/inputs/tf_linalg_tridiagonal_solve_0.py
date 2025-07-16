
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_tridiagonal_solve_inputs():
    list_of_inputs = []

    # Input 1
    diagonals = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]], dtype=np.float32)
    rhs = np.array([[10.0, 11.0, 12.0]], dtype=np.float32)
    input_dict = {'diagonals': diagonals, 'rhs': rhs, 'diagonals_format': 'compact', 'transpose_rhs': False, 'conjugate_rhs': False, 'name': 'test1', 'partial_pivoting': True, 'perturb_singular': False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    diagonals = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]], dtype=np.float64)
    rhs = np.array([[10.0, 11.0, 12.0]], dtype=np.float64)
    input_dict = {'diagonals': diagonals, 'rhs': rhs, 'diagonals_format': 'compact', 'transpose_rhs': False, 'conjugate_rhs': False, 'name': 'test2', 'partial_pivoting': False, 'perturb_singular': False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    diagonals = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], dtype=np.complex64)
    rhs = np.array([[7.0+1j, 8.0+2j]], dtype=np.complex64)
    input_dict = {'diagonals': diagonals, 'rhs': rhs, 'diagonals_format': 'compact', 'transpose_rhs': True, 'conjugate_rhs': True, 'name': 'test3', 'partial_pivoting': True, 'perturb_singular': False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    diagonals = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], dtype=np.complex128)
    rhs = np.array([[7.0+1j, 8.0+2j]], dtype=np.complex128)
    input_dict = {'diagonals': diagonals, 'rhs': rhs, 'diagonals_format': 'compact', 'transpose_rhs': True, 'conjugate_rhs': False, 'name': 'test4', 'partial_pivoting': False, 'perturb_singular': False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    diagonals = np.array([[[1.0, 0.0, 0.0], [2.0, 3.0, 0.0], [0.0, 4.0, 5.0]]], dtype=np.float32)
    rhs = np.array([[6.0, 7.0, 8.0]], dtype=np.float32)
    input_dict = {'diagonals': diagonals, 'rhs': rhs, 'diagonals_format': 'matrix', 'transpose_rhs': False, 'conjugate_rhs': False, 'name': 'test5', 'partial_pivoting': True, 'perturb_singular': False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    sup = np.array([1.0, 2.0], dtype=np.float32)
    main = np.array([3.0, 4.0, 5.0], dtype=np.float32)
    sub = np.array([6.0, 7.0], dtype=np.float32)
    rhs = np.array([[8.0, 9.0, 10.0]], dtype=np.float32)
    diagonals = (sup, main, sub)
    input_dict = {'diagonals': diagonals, 'rhs': rhs, 'diagonals_format': 'sequence', 'transpose_rhs': False, 'conjugate_rhs': False, 'name': 'test6', 'partial_pivoting': True, 'perturb_singular': False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    diagonals = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]], dtype=np.float32)
    rhs = np.array([[10.0, 11.0, 12.0]], dtype=np.float32)
    input_dict = {'diagonals': diagonals, 'rhs': rhs, 'diagonals_format': 'compact', 'transpose_rhs': False, 'conjugate_rhs': False, 'name': 'test7', 'partial_pivoting': True, 'perturb_singular': False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    diagonals = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]], dtype=np.float32)
    rhs = np.array([[10.0, 11.0, 12.0]], dtype=np.float32)
    input_dict = {'diagonals': diagonals, 'rhs': rhs, 'diagonals_format': 'compact', 'transpose_rhs': False, 'conjugate_rhs': False, 'name': 'test8', 'partial_pivoting': True, 'perturb_singular': True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Complex64 with perturb_singular
    diagonals = np.array([[[1.0+1j, 2.0+2j, 3.0+3j], [4.0+4j, 5.0+5j, 6.0+6j], [7.0+7j, 8.0+8j, 9.0+9j]]], dtype=np.complex64)
    rhs = np.array([[10.0+10j, 11.0+11j, 12.0+12j]], dtype=np.complex64)
    input_dict = {'diagonals': diagonals, 'rhs': rhs, 'diagonals_format': 'compact', 'transpose_rhs': False, 'conjugate_rhs': False, 'name': 'test9', 'partial_pivoting': True, 'perturb_singular': False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: solving multiple RHS
    diagonals = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]], dtype=np.float32)
    rhs = np.array([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]], dtype=np.float32)
    input_dict = {'diagonals': diagonals, 'rhs': rhs, 'diagonals_format': 'compact', 'transpose_rhs': False, 'conjugate_rhs': False, 'name': 'test10', 'partial_pivoting': True, 'perturb_singular': False}
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
