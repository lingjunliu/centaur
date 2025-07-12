
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_tridiagonal_solve_inputs():
    list_of_inputs = []

    # Input 1: compact format, simple case
    diagonals = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]], dtype=np.float32)
    rhs = np.array([[10.0, 11.0, 12.0]], dtype=np.float32)
    input_dict = {'diagonals': diagonals, 'rhs': rhs, 'diagonals_format': 'compact', 'transpose_rhs': False, 'conjugate_rhs': False, 'name': None, 'partial_pivoting': True, 'perturb_singular': False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: matrix format
    matrix = np.array([[[1.0, 2.0, 0.0], [4.0, 5.0, 6.0], [0.0, 8.0, 9.0]]], dtype=np.float32)
    rhs = np.array([[10.0, 11.0, 12.0]], dtype=np.float32)
    input_dict = {'diagonals': matrix, 'rhs': rhs, 'diagonals_format': 'matrix', 'transpose_rhs': False, 'conjugate_rhs': False, 'name': None, 'partial_pivoting': True, 'perturb_singular': False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: multiple RHS
    diagonals = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]], dtype=np.float32)
    rhs = np.array([[[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]]], dtype=np.float32)
    input_dict = {'diagonals': diagonals, 'rhs': rhs, 'diagonals_format': 'compact', 'transpose_rhs': False, 'conjugate_rhs': False, 'name': None, 'partial_pivoting': True, 'perturb_singular': False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: transpose_rhs=True
    diagonals = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]], dtype=np.float32)
    rhs = np.array([[10.0, 11.0, 12.0]], dtype=np.float32)
    input_dict = {'diagonals': diagonals, 'rhs': rhs, 'diagonals_format': 'compact', 'transpose_rhs': True, 'conjugate_rhs': False, 'name': None, 'partial_pivoting': True, 'perturb_singular': False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex numbers
    diagonals = np.array([[[1.0j, 2.0j, 3.0j], [4.0j, 5.0j, 6.0j], [7.0j, 8.0j, 9.0j]]], dtype=np.complex64)
    rhs = np.array([[10.0j, 11.0j, 12.0j]], dtype=np.complex64)
    input_dict = {'diagonals': diagonals, 'rhs': rhs, 'diagonals_format': 'compact', 'transpose_rhs': False, 'conjugate_rhs': False, 'name': None, 'partial_pivoting': True, 'perturb_singular': False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: conjugate_rhs=True
    diagonals = np.array([[[1.0j, 2.0j, 3.0j], [4.0j, 5.0j, 6.0j], [7.0j, 8.0j, 9.0j]]], dtype=np.complex64)
    rhs = np.array([[10.0j, 11.0j, 12.0j]], dtype=np.complex64)
    input_dict = {'diagonals': diagonals, 'rhs': rhs, 'diagonals_format': 'compact', 'transpose_rhs': False, 'conjugate_rhs': True, 'name': None, 'partial_pivoting': True, 'perturb_singular': False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: partial_pivoting=False
    diagonals = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]], dtype=np.float32)
    rhs = np.array([[10.0, 11.0, 12.0]], dtype=np.float32)
    input_dict = {'diagonals': diagonals, 'rhs': rhs, 'diagonals_format': 'compact', 'transpose_rhs': False, 'conjugate_rhs': False, 'name': None, 'partial_pivoting': False, 'perturb_singular': False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: perturb_singular=True
    diagonals = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=np.float32)
    rhs = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
    input_dict = {'diagonals': diagonals, 'rhs': rhs, 'diagonals_format': 'compact', 'transpose_rhs': False, 'conjugate_rhs': False, 'name': None, 'partial_pivoting': True, 'perturb_singular': True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: name
    diagonals = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]], dtype=np.float32)
    rhs = np.array([[10.0, 11.0, 12.0]], dtype=np.float32)
    input_dict = {'diagonals': diagonals, 'rhs': rhs, 'diagonals_format': 'compact', 'transpose_rhs': False, 'conjugate_rhs': False, 'name': "test_solve", 'partial_pivoting': True, 'perturb_singular': False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: higher dimensions, compact format
    diagonals = np.array([[[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]]], dtype=np.float32)
    rhs = np.array([[[[10.0, 11.0, 12.0]]]] , dtype=np.float32)
    input_dict = {'diagonals': diagonals, 'rhs': rhs, 'diagonals_format': 'compact', 'transpose_rhs': False, 'conjugate_rhs': False, 'name': None, 'partial_pivoting': True, 'perturb_singular': False}
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
