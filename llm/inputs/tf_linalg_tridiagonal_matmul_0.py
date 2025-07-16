
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_tridiagonal_matmul_inputs():
    list_of_inputs = []

    # Input 1: Compact format, simple case
    diagonals = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=np.float32)
    rhs = np.array([[[1, 1], [1, 1], [1, 1]]], dtype=np.float32)
    diagonals_format = 'compact'
    name = 'matmul1'
    input_dict = {'diagonals': tf.convert_to_tensor(diagonals), 'rhs': tf.convert_to_tensor(rhs), 'diagonals_format': diagonals_format, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Sequence format, simple case
    superdiag = np.array([-1, -1, 0], dtype=np.float64)
    maindiag = np.array([2, 2, 2], dtype=np.float64)
    subdiag = np.array([0, -1, -1], dtype=np.float64)
    rhs = np.array([[1, 1], [1, 1], [1, 1]], dtype=np.float64)
    diagonals_format = 'sequence'
    name = 'matmul2'
    input_dict = {'diagonals': (tf.convert_to_tensor(superdiag), tf.convert_to_tensor(maindiag), tf.convert_to_tensor(subdiag)), 'rhs': tf.convert_to_tensor(rhs), 'diagonals_format': diagonals_format, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Compact format, with batch dimension
    diagonals = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[9, 8, 7], [6, 5, 4], [3, 2, 1]]], dtype=np.float32)
    rhs = np.array([[[1, 1], [1, 1], [1, 1]], [[2, 2], [2, 2], [2, 2]]], dtype=np.float32)
    diagonals_format = 'compact'
    name = 'matmul3'
    input_dict = {'diagonals': tf.convert_to_tensor(diagonals), 'rhs': tf.convert_to_tensor(rhs), 'diagonals_format': diagonals_format, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Sequence format, with batch dimension
    superdiag = np.array([[1, 2, 3], [4, 5, 0]], dtype=np.float64)
    maindiag = np.array([[4, 5, 6], [7, 8, 9]], dtype=np.float64)
    subdiag = np.array([[7, 8, 0], [1, 2, 3]], dtype=np.float64)
    rhs = np.array([[[1, 1], [1, 1], [1, 1]], [[2, 2], [2, 2], [2, 2]]], dtype=np.float64)
    diagonals_format = 'sequence'
    name = 'matmul4'
    input_dict = {'diagonals': (tf.convert_to_tensor(superdiag), tf.convert_to_tensor(maindiag), tf.convert_to_tensor(subdiag)), 'rhs': tf.convert_to_tensor(rhs), 'diagonals_format': diagonals_format, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Compact format, complex numbers
    diagonals = np.array([[[1+1j, 2+2j, 3+3j], [4+4j, 5+5j, 6+6j], [7+7j, 8+8j, 9+9j]]], dtype=np.complex64)
    rhs = np.array([[[1+1j, 1+1j], [1+1j, 1+1j], [1+1j, 1+1j]]], dtype=np.complex64)
    diagonals_format = 'compact'
    name = 'matmul5'
    input_dict = {'diagonals': tf.convert_to_tensor(diagonals), 'rhs': tf.convert_to_tensor(rhs), 'diagonals_format': diagonals_format, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Sequence format, complex numbers
    superdiag = np.array([-1j, -1j, 0], dtype=np.complex128)
    maindiag = np.array([2, 2j, 2], dtype=np.complex128)
    subdiag = np.array([0, -1, -1j], dtype=np.complex128)
    rhs = np.array([[1, 1], [1, 1j], [1j, 1]], dtype=np.complex128)
    diagonals_format = 'sequence'
    name = 'matmul6'
    input_dict = {'diagonals': (tf.convert_to_tensor(superdiag), tf.convert_to_tensor(maindiag), tf.convert_to_tensor(subdiag)), 'rhs': tf.convert_to_tensor(rhs), 'diagonals_format': diagonals_format, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7: Compact format, different rhs shape
    diagonals = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=np.float32)
    rhs = np.array([[[1], [1], [1]]], dtype=np.float32)
    diagonals_format = 'compact'
    name = 'matmul7'
    input_dict = {'diagonals': tf.convert_to_tensor(diagonals), 'rhs': tf.convert_to_tensor(rhs), 'diagonals_format': diagonals_format, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Sequence format, different rhs shape
    superdiag = np.array([-1, -1, 0], dtype=np.float64)
    maindiag = np.array([2, 2, 2], dtype=np.float64)
    subdiag = np.array([0, -1, -1], dtype=np.float64)
    rhs = np.array([[1], [1], [1]], dtype=np.float64)
    diagonals_format = 'sequence'
    name = 'matmul8'
    input_dict = {'diagonals': (tf.convert_to_tensor(superdiag), tf.convert_to_tensor(maindiag), tf.convert_to_tensor(subdiag)), 'rhs': tf.convert_to_tensor(rhs), 'diagonals_format': diagonals_format, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Compact format, floats with negatives
    diagonals = np.array([[[1.5, -2.5, 3.5], [-4.5, 5.5, -6.5], [7.5, -8.5, 9.5]]], dtype=np.float32)
    rhs = np.array([[[1, -1], [-1, 1], [1, -1]]], dtype=np.float32)
    diagonals_format = 'compact'
    name = 'matmul9'
    input_dict = {'diagonals': tf.convert_to_tensor(diagonals), 'rhs': tf.convert_to_tensor(rhs), 'diagonals_format': diagonals_format, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Sequence format, floats with negatives
    superdiag = np.array([-1.2, -1.3, 0], dtype=np.float64)
    maindiag = np.array([2.4, -2.5, 2.6], dtype=np.float64)
    subdiag = np.array([0, -1.7, -1.8], dtype=np.float64)
    rhs = np.array([[1.1, -1.2], [-1.3, 1.4], [1.5, -1.6]], dtype=np.float64)
    diagonals_format = 'sequence'
    name = 'matmul10'
    input_dict = {'diagonals': (tf.convert_to_tensor(superdiag), tf.convert_to_tensor(maindiag), tf.convert_to_tensor(subdiag)), 'rhs': tf.convert_to_tensor(rhs), 'diagonals_format': diagonals_format, 'name': name}
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
