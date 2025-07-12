
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
    diagonals = (tf.convert_to_tensor(superdiag), tf.convert_to_tensor(maindiag), tf.convert_to_tensor(subdiag))
    rhs = tf.convert_to_tensor(np.array([[1, 1], [1, 1], [1, 1]], dtype=np.float32))
    diagonals_format = 'sequence'
    name = None

    input_dict = {
        "diagonals": diagonals,
        "rhs": rhs,
        "diagonals_format": diagonals_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    diagonals = tf.convert_to_tensor(np.array([[-1, -1, 0], [2, 2, 2], [0, -1, -1]], dtype=np.float64))
    rhs = tf.convert_to_tensor(np.array([[1, 1], [1, 1], [1, 1]], dtype=np.float64))
    diagonals_format = 'compact'
    name = "matmul_op"

    input_dict = {
        "diagonals": diagonals,
        "rhs": rhs,
        "diagonals_format": diagonals_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    superdiag = np.array([-1j, -1j, 0j], dtype=np.complex64)
    maindiag = np.array([2j, 2j, 2j], dtype=np.complex64)
    subdiag = np.array([0j, -1j, -1j], dtype=np.complex64)
    diagonals = (tf.convert_to_tensor(superdiag), tf.convert_to_tensor(maindiag), tf.convert_to_tensor(subdiag))
    rhs = tf.convert_to_tensor(np.array([[1j, 1j], [1j, 1j], [1j, 1j]], dtype=np.complex64))
    diagonals_format = 'sequence'
    name = None

    input_dict = {
        "diagonals": diagonals,
        "rhs": rhs,
        "diagonals_format": diagonals_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    diagonals = tf.convert_to_tensor(np.array([[-1j, -1j, 0j], [2j, 2j, 2j], [0j, -1j, -1j]], dtype=np.complex128))
    rhs = tf.convert_to_tensor(np.array([[1j, 1j], [1j, 1j], [1j, 1j]], dtype=np.complex128))
    diagonals_format = 'compact'
    name = None

    input_dict = {
        "diagonals": diagonals,
        "rhs": rhs,
        "diagonals_format": diagonals_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5: Batched example
    superdiag = np.array([[-1, -1, 0], [-2, -2, 0]], dtype=np.float32)
    maindiag = np.array([[2, 2, 2], [3, 3, 3]], dtype=np.float32)
    subdiag = np.array([[0, -1, -1], [0, -2, -2]], dtype=np.float32)
    diagonals = (tf.convert_to_tensor(superdiag), tf.convert_to_tensor(maindiag), tf.convert_to_tensor(subdiag))
    rhs = tf.convert_to_tensor(np.array([[[1, 1], [1, 1], [1, 1]], [[2, 2], [2, 2], [2, 2]]], dtype=np.float32))
    diagonals_format = 'sequence'
    name = None

    input_dict = {
        "diagonals": diagonals,
        "rhs": rhs,
        "diagonals_format": diagonals_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Batched compact format
    diagonals = tf.convert_to_tensor(np.array([
        [[-1, -1, 0], [2, 2, 2], [0, -1, -1]],
        [[-2, -2, 0], [3, 3, 3], [0, -2, -2]]
    ], dtype=np.float64))
    rhs = tf.convert_to_tensor(np.array([[[1, 1], [1, 1], [1, 1]], [[2, 2], [2, 2], [2, 2]]], dtype=np.float64))
    diagonals_format = 'compact'
    name = "batch_matmul"
    input_dict = {
        "diagonals": diagonals,
        "rhs": rhs,
        "diagonals_format": diagonals_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger matrix
    superdiag = np.array([-1, -1, -1, 0], dtype=np.float32)
    maindiag = np.array([2, 2, 2, 2], dtype=np.float32)
    subdiag = np.array([0, -1, -1, -1], dtype=np.float32)
    diagonals = (tf.convert_to_tensor(superdiag), tf.convert_to_tensor(maindiag), tf.convert_to_tensor(subdiag))
    rhs = tf.convert_to_tensor(np.array([[1, 1], [1, 1], [1, 1], [1, 1]], dtype=np.float32))
    diagonals_format = 'sequence'
    name = None
    input_dict = {
        "diagonals": diagonals,
        "rhs": rhs,
        "diagonals_format": diagonals_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: compact format, larger matrix
    diagonals = tf.convert_to_tensor(np.array([[-1, -1, -1, 0], [2, 2, 2, 2], [0, -1, -1, -1]], dtype=np.float64))
    rhs = tf.convert_to_tensor(np.array([[1, 1], [1, 1], [1, 1], [1, 1]], dtype=np.float64))
    diagonals_format = 'compact'
    name = "large_compact"
    input_dict = {
        "diagonals": diagonals,
        "rhs": rhs,
        "diagonals_format": diagonals_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: rhs with different N
    superdiag = np.array([-1, -1, 0], dtype=np.float32)
    maindiag = np.array([2, 2, 2], dtype=np.float32)
    subdiag = np.array([0, -1, -1], dtype=np.float32)
    diagonals = (tf.convert_to_tensor(superdiag), tf.convert_to_tensor(maindiag), tf.convert_to_tensor(subdiag))
    rhs = tf.convert_to_tensor(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32))
    diagonals_format = 'sequence'
    name = None
    input_dict = {
        "diagonals": diagonals,
        "rhs": rhs,
        "diagonals_format": diagonals_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: batched sequence with different N
    superdiag = np.array([[-1, -1, 0], [-2, -2, 0]], dtype=np.float32)
    maindiag = np.array([[2, 2, 2], [3, 3, 3]], dtype=np.float32)
    subdiag = np.array([[0, -1, -1], [0, -2, -2]], dtype=np.float32)
    diagonals = (tf.convert_to_tensor(superdiag), tf.convert_to_tensor(maindiag), tf.convert_to_tensor(subdiag))
    rhs = tf.convert_to_tensor(np.array([[[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[2, 2, 2], [2, 2, 2], [2, 2, 2]]], dtype=np.float32))
    diagonals_format = 'sequence'
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
