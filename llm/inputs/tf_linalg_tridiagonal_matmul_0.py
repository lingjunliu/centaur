
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_tridiagonal_matmul_inputs():
    list_of_inputs = []

    # Input 1
    superdiag = np.array([-1, -1, 0], dtype=np.float64)
    maindiag = np.array([2, 2, 2], dtype=np.float64)
    subdiag = np.array([0, -1, -1], dtype=np.float64)
    diagonals = (tf.convert_to_tensor(superdiag), tf.convert_to_tensor(maindiag), tf.convert_to_tensor(subdiag))
    rhs = tf.convert_to_tensor(np.array([[1, 1], [1, 1], [1, 1]], dtype=np.float64))
    input_dict = {"diagonals": diagonals, "rhs": rhs, "diagonals_format": 'sequence', "name": "matmul_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    diagonals = tf.convert_to_tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32))
    rhs = tf.convert_to_tensor(np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32))
    input_dict = {"diagonals": diagonals, "rhs": rhs, "diagonals_format": 'compact', "name": "matmul_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    superdiag = np.array([-1, -2, 0, 1], dtype=np.float64)
    maindiag = np.array([2, 3, 4, 5], dtype=np.float64)
    subdiag = np.array([0, -1, -2, -3], dtype=np.float64)
    diagonals = (tf.convert_to_tensor(superdiag), tf.convert_to_tensor(maindiag), tf.convert_to_tensor(subdiag))
    rhs = tf.convert_to_tensor(np.array([[1, 1], [1, 1], [1, 1], [1,1]], dtype=np.float64))
    input_dict = {"diagonals": diagonals, "rhs": rhs, "diagonals_format": 'sequence', "name": "matmul_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    diagonals = tf.convert_to_tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float32))
    rhs = tf.convert_to_tensor(np.array([[1, 2], [3, 4], [5, 6], [7,8]], dtype=np.float32))
    input_dict = {"diagonals": diagonals, "rhs": rhs, "diagonals_format": 'compact', "name": "matmul_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    superdiag = np.array([-1j, -2j, 0], dtype=np.complex128)
    maindiag = np.array([2j, 3j, 4j], dtype=np.complex128)
    subdiag = np.array([0, -1j, -2j], dtype=np.complex128)
    diagonals = (tf.convert_to_tensor(superdiag), tf.convert_to_tensor(maindiag), tf.convert_to_tensor(subdiag))
    rhs = tf.convert_to_tensor(np.array([[1j, 1j], [1j, 1j], [1j, 1j]], dtype=np.complex128))
    input_dict = {"diagonals": diagonals, "rhs": rhs, "diagonals_format": 'sequence', "name": "matmul_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    diagonals = tf.convert_to_tensor(np.array([[1j, 2j, 3j], [4j, 5j, 6j], [7j, 8j, 9j]], dtype=np.complex64))
    rhs = tf.convert_to_tensor(np.array([[1j, 2j], [3j, 4j], [5j, 6j]], dtype=np.complex64))
    input_dict = {"diagonals": diagonals, "rhs": rhs, "diagonals_format": 'compact', "name": "matmul_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 (batch)
    superdiag = np.array([[-1, -1, 0], [-2, -2, 0]], dtype=np.float64)
    maindiag = np.array([[2, 2, 2], [3, 3, 3]], dtype=np.float64)
    subdiag = np.array([[0, -1, -1], [0, -2, -2]], dtype=np.float64)
    diagonals = (tf.convert_to_tensor(superdiag), tf.convert_to_tensor(maindiag), tf.convert_to_tensor(subdiag))
    rhs = tf.convert_to_tensor(np.array([[[1, 1], [1, 1], [1, 1]], [[2, 2], [2, 2], [2, 2]]], dtype=np.float64))
    input_dict = {"diagonals": diagonals, "rhs": rhs, "diagonals_format": 'sequence', "name": "matmul_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 (batch compact)
    diagonals = tf.convert_to_tensor(np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]]], dtype=np.float32))
    rhs = tf.convert_to_tensor(np.array([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]], dtype=np.float32))
    input_dict = {"diagonals": diagonals, "rhs": rhs, "diagonals_format": 'compact', "name": "matmul_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    superdiag = np.array([[-1, -2, 0, 1],[-3, -4, 0, 2]], dtype=np.float64)
    maindiag = np.array([[2, 3, 4, 5],[6,7,8,9]], dtype=np.float64)
    subdiag = np.array([[0, -1, -2, -3],[0,-5,-6,-7]], dtype=np.float64)
    diagonals = (tf.convert_to_tensor(superdiag), tf.convert_to_tensor(maindiag), tf.convert_to_tensor(subdiag))
    rhs = tf.convert_to_tensor(np.array([[[1, 1], [1, 1], [1, 1], [1,1]],[[2, 2], [2, 2], [2, 2], [2,2]]], dtype=np.float64))
    input_dict = {"diagonals": diagonals, "rhs": rhs, "diagonals_format": 'sequence', "name": "matmul_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 (complex64 sequence, different shape)
    superdiag = np.array([-1j, -2j, 0], dtype=np.complex64)
    maindiag = np.array([2j, 3j, 4j], dtype=np.complex64)
    subdiag = np.array([0, -1j, -2j], dtype=np.complex64)
    diagonals = (tf.convert_to_tensor(superdiag), tf.convert_to_tensor(maindiag), tf.convert_to_tensor(subdiag))
    rhs = tf.convert_to_tensor(np.array([[1j], [1j], [1j]], dtype=np.complex64))
    input_dict = {"diagonals": diagonals, "rhs": rhs, "diagonals_format": 'sequence', "name": "matmul_10"}
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
