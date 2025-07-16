
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_tridiagonal_matmul_inputs():
    list_of_inputs = []

    # Input 1
    superdiag = np.array([-1, -1, 0], dtype=np.float32)
    maindiag = np.array([2, 2, 2], dtype=np.float32)
    subdiag = np.array([0, -1, -1], dtype=np.float32)
    rhs = np.array([[1, 1], [1, 1], [1, 1]], dtype=np.float32)
    diagonals_format = 'sequence'
    name = None

    input_dict = {
        "diagonals": (superdiag, maindiag, subdiag),
        "rhs": rhs,
        "diagonals_format": diagonals_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    diagonals = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=np.float64)
    rhs = np.array([[[1, 2], [3, 4], [5, 6]]], dtype=np.float64)
    diagonals_format = 'matrix'
    name = "test_matmul"

    input_dict = {
        "diagonals": diagonals,
        "rhs": rhs,
        "diagonals_format": diagonals_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    diagonals = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=np.complex64)
    rhs = np.array([[[1, 2], [3, 4], [5, 6]]], dtype=np.complex64)
    diagonals_format = 'matrix'
    name = None

    input_dict = {
        "diagonals": diagonals,
        "rhs": rhs,
        "diagonals_format": diagonals_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    diagonals = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=np.complex128)
    rhs = np.array([[[1, 2], [3, 4], [5, 6]]], dtype=np.complex128)
    diagonals_format = 'matrix'
    name = None

    input_dict = {
        "diagonals": diagonals,
        "rhs": rhs,
        "diagonals_format": diagonals_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    diagonals = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=np.float32)
    rhs = np.array([[[1, 2], [3, 4], [5, 6]]], dtype=np.float32)
    diagonals_format = 'matrix'
    name = None

    input_dict = {
        "diagonals": diagonals,
        "rhs": rhs,
        "diagonals_format": diagonals_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    superdiag = np.array([-1, -1, 0], dtype=np.float64)
    maindiag = np.array([2, 2, 2], dtype=np.float64)
    subdiag = np.array([0, -1, -1], dtype=np.float64)
    rhs = np.array([[1, 1], [1, 1], [1, 1]], dtype=np.float64)
    diagonals_format = 'sequence'
    name = None

    input_dict = {
        "diagonals": (superdiag, maindiag, subdiag),
        "rhs": rhs,
        "diagonals_format": diagonals_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    diagonals = np.random.rand(3, 3, 3).astype(np.float32)
    rhs = np.random.rand(3, 3, 4).astype(np.float32)
    diagonals_format = "matrix"
    name = "random_matrix"

    input_dict = {
        "diagonals": diagonals,
        "rhs": rhs,
        "diagonals_format": diagonals_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    superdiag = np.array([-1, -1, 0], dtype=np.complex128)
    maindiag = np.array([2, 2, 2], dtype=np.complex128)
    subdiag = np.array([0, -1, -1], dtype=np.complex128)
    rhs = np.array([[1+1j, 1+1j], [1+1j, 1+1j], [1+1j, 1+1j]], dtype=np.complex128)
    diagonals_format = 'sequence'
    name = "complex_sequence"

    input_dict = {
        "diagonals": (superdiag, maindiag, subdiag),
        "rhs": rhs,
        "diagonals_format": diagonals_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    diagonals = np.array([[[1, 2, 3], [4, 5, 6], [7,8,9]]], dtype=np.float32)
    rhs = np.array([[[1, 2], [3, 4], [5,6]]], dtype=np.float32)
    diagonals_format = 'compact'
    name = None

    input_dict = {
        "diagonals": diagonals,
        "rhs": rhs,
        "diagonals_format": diagonals_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    diagonals = np.array([[[1, 2, 3], [4, 5, 6], [7,8,9]]], dtype=np.float64)
    rhs = np.array([[[1, 2], [3, 4], [5,6]]], dtype=np.float64)
    diagonals_format = 'compact'
    name = None

    input_dict = {
        "diagonals": diagonals,
        "rhs": rhs,
        "diagonals_format": diagonals_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
     # Input 11
    diagonals = np.array([[[1, 2, 3], [4, 5, 6], [7,8,9]]], dtype=np.complex64)
    rhs = np.array([[[1, 2], [3, 4], [5,6]]], dtype=np.complex64)
    diagonals_format = 'compact'
    name = None

    input_dict = {
        "diagonals": diagonals,
        "rhs": rhs,
        "diagonals_format": diagonals_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    diagonals = np.random.rand(3, 3, 3).astype(np.float32)
    rhs = np.random.rand(3, 3, 4).astype(np.float32)
    diagonals_format = 'compact'
    name = None

    input_dict = {
        "diagonals": diagonals,
        "rhs": rhs,
        "diagonals_format": diagonals_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13
    diagonals = np.array([[[1, 2], [3, 4], [5, 6]],[[7, 8], [9, 10], [11, 12]]], dtype=np.float32)
    rhs = np.array([[[1, 1], [1, 1], [1, 1]],[[2, 2], [2, 2], [2, 2]]], dtype=np.float32)
    diagonals_format = 'compact'
    name = None

    input_dict = {
        "diagonals": diagonals,
        "rhs": rhs,
        "diagonals_format": diagonals_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.tridiagonal_matmul"] = tf_linalg_tridiagonal_matmul_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.tridiagonal_matmul' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.tridiagonal_matmul'.")

check_valid('tf.linalg.tridiagonal_matmul', generated_inputs['tf.linalg.tridiagonal_matmul'], lib="tf", suffix=0)
