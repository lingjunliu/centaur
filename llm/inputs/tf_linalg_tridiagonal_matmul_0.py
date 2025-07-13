
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_tridiagonal_matmul_inputs():
    list_of_inputs = []

    # Input 1: Basic case with compact format
    diagonals = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]], dtype=np.float32)
    rhs = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], dtype=np.float32)
    diagonals_format = 'compact'
    name = 'matmul_1'
    input_dict = {'diagonals': tf.convert_to_tensor(diagonals), 'rhs': tf.convert_to_tensor(rhs), 'diagonals_format': diagonals_format, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Sequence format
    superdiag = np.array([1.0, 2.0], dtype=np.float32)
    maindiag = np.array([3.0, 4.0, 5.0], dtype=np.float32)
    subdiag = np.array([6.0, 7.0], dtype=np.float32)
    rhs = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    diagonals_format = 'sequence'
    name = 'matmul_2'
    input_dict = {'diagonals': (tf.convert_to_tensor(superdiag), tf.convert_to_tensor(maindiag), tf.convert_to_tensor(subdiag)), 'rhs': tf.convert_to_tensor(rhs), 'diagonals_format': diagonals_format, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Compact format, different dtype
    diagonals = np.array([[[1 + 1j, 2 + 2j, 3 + 3j], [4 + 4j, 5 + 5j, 6 + 6j], [7 + 7j, 8 + 8j, 9 + 9j]]], dtype=np.complex64)
    rhs = np.array([[[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j], [5 + 5j, 6 + 6j]]], dtype=np.complex64)
    diagonals_format = 'compact'
    name = 'matmul_3'
    input_dict = {'diagonals': tf.convert_to_tensor(diagonals), 'rhs': tf.convert_to_tensor(rhs), 'diagonals_format': diagonals_format, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Sequence format, different dtype
    superdiag = np.array([1.0 + 1j, 2.0 + 2j], dtype=np.complex64)
    maindiag = np.array([3.0 + 3j, 4.0 + 4j, 5.0 + 5j], dtype=np.complex64)
    subdiag = np.array([6.0 + 6j, 7.0 + 7j], dtype=np.complex64)
    rhs = np.array([[1.0 + 1j, 2.0 + 2j], [3.0 + 3j, 4.0 + 4j], [5.0 + 5j, 6.0 + 6j]], dtype=np.complex64)
    diagonals_format = 'sequence'
    name = 'matmul_4'
    input_dict = {'diagonals': (tf.convert_to_tensor(superdiag), tf.convert_to_tensor(maindiag), tf.convert_to_tensor(subdiag)), 'rhs': tf.convert_to_tensor(rhs), 'diagonals_format': diagonals_format, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 5: Compact format, multiple batches
    diagonals = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]], dtype=np.float32)
    rhs = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]], dtype=np.float32)
    diagonals_format = 'compact'
    name = 'matmul_5'
    input_dict = {'diagonals': tf.convert_to_tensor(diagonals), 'rhs': tf.convert_to_tensor(rhs), 'diagonals_format': diagonals_format, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Sequence format, multiple batches
    superdiag = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    maindiag = np.array([[3.0, 4.0, 5.0], [5.0, 6.0, 7.0]], dtype=np.float32)
    subdiag = np.array([[6.0, 7.0], [8.0, 9.0]], dtype=np.float32)
    rhs = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]], dtype=np.float32)
    diagonals_format = 'sequence'
    name = 'matmul_6'
    input_dict = {'diagonals': (tf.convert_to_tensor(superdiag), tf.convert_to_tensor(maindiag), tf.convert_to_tensor(subdiag)), 'rhs': tf.convert_to_tensor(rhs), 'diagonals_format': diagonals_format, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Compact format, larger matrix
    diagonals = np.array([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]], dtype=np.float32)
    rhs = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    diagonals_format = 'compact'
    name = 'matmul_7'
    input_dict = {'diagonals': tf.convert_to_tensor(diagonals), 'rhs': tf.convert_to_tensor(rhs), 'diagonals_format': diagonals_format, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Sequence format, larger matrix
    superdiag = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    maindiag = np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32)
    subdiag = np.array([8.0, 9.0, 10.0], dtype=np.float32)
    rhs = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    diagonals_format = 'sequence'
    name = 'matmul_8'
    input_dict = {'diagonals': (tf.convert_to_tensor(superdiag), tf.convert_to_tensor(maindiag), tf.convert_to_tensor(subdiag)), 'rhs': tf.convert_to_tensor(rhs), 'diagonals_format': diagonals_format, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Compact format, negative values
    diagonals = np.array([[[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0], [7.0, -8.0, 9.0]]], dtype=np.float32)
    rhs = np.array([[[1.0, -2.0], [-3.0, 4.0], [5.0, -6.0]]], dtype=np.float32)
    diagonals_format = 'compact'
    name = 'matmul_9'
    input_dict = {'diagonals': tf.convert_to_tensor(diagonals), 'rhs': tf.convert_to_tensor(rhs), 'diagonals_format': diagonals_format, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Sequence format, negative values
    superdiag = np.array([1.0, -2.0], dtype=np.float32)
    maindiag = np.array([-3.0, 4.0, -5.0], dtype=np.float32)
    subdiag = np.array([-6.0, 7.0], dtype=np.float32)
    rhs = np.array([[1.0, -2.0], [-3.0, 4.0], [5.0, -6.0]], dtype=np.float32)
    diagonals_format = 'sequence'
    name = 'matmul_10'
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
