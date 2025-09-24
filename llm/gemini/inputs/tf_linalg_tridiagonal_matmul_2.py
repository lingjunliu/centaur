
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
    diagonals = (superdiag, maindiag, subdiag)
    rhs = np.array([[1, 1], [1, 1], [1, 1]], dtype=np.float32)
    diagonals_format = 'sequence'
    name = 'test1'
    input_dict = {'diagonals': diagonals, 'rhs': rhs, 'diagonals_format': diagonals_format, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    diagonals = np.array([[[1,2,3],[4,5,6],[7,8,9]]], dtype=np.float64)
    rhs = np.array([[[1, 1], [1, 1], [1, 1]]], dtype=np.float64)
    diagonals_format = 'compact'
    name = 'test2'
    input_dict = {'diagonals': diagonals, 'rhs': rhs, 'diagonals_format': diagonals_format, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    superdiag = np.array([-1, -1, 0, 1], dtype=np.complex64)
    maindiag = np.array([2, 2, 2, 3], dtype=np.complex64)
    subdiag = np.array([0, -1, -1, 2], dtype=np.complex64)
    diagonals = (superdiag, maindiag, subdiag)
    rhs = np.array([[1j, 1], [1, 1j], [1, 1], [1j, 1j]], dtype=np.complex64)
    diagonals_format = 'sequence'
    name = 'test3'
    input_dict = {'diagonals': diagonals, 'rhs': rhs, 'diagonals_format': diagonals_format, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    diagonals = np.array([[[1,2],[3,4],[5,6]]], dtype=np.float32)
    rhs = np.array([[[1, 1], [1, 1]]], dtype=np.float32)
    diagonals_format = 'compact'
    name = 'test4'
    input_dict = {'diagonals': diagonals, 'rhs': rhs, 'diagonals_format': diagonals_format, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    superdiag = np.array([-1, -1], dtype=np.float32)
    maindiag = np.array([2, 2], dtype=np.float32)
    subdiag = np.array([0, -1], dtype=np.float32)
    diagonals = (superdiag, maindiag, subdiag)
    rhs = np.array([[1, 1], [1, 1]], dtype=np.float32)
    diagonals_format = 'sequence'
    name = 'test5'
    input_dict = {'diagonals': diagonals, 'rhs': rhs, 'diagonals_format': diagonals_format, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    diagonals = np.array([[[1,2],[3,4],[5,6]],[[7,8],[9,10],[11,12]]], dtype=np.float64)
    rhs = np.array([[[1, 1], [1, 1]],[[2,2],[2,2]]], dtype=np.float64)
    diagonals_format = 'compact'
    name = 'test6'
    input_dict = {'diagonals': diagonals, 'rhs': rhs, 'diagonals_format': diagonals_format, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    superdiag = np.array([-1, -1, 0], dtype=np.complex128)
    maindiag = np.array([2, 2, 2], dtype=np.complex128)
    subdiag = np.array([0, -1, -1], dtype=np.complex128)
    diagonals = (superdiag, maindiag, subdiag)
    rhs = np.array([[1j, 1], [1, 1j], [1, 1]], dtype=np.complex128)
    diagonals_format = 'sequence'
    name = 'test7'
    input_dict = {'diagonals': diagonals, 'rhs': rhs, 'diagonals_format': diagonals_format, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8
    superdiag = np.array([[-1, -1, 0], [-2, -2, 1]], dtype=np.float32)
    maindiag = np.array([[2, 2, 2], [3, 3, 3]], dtype=np.float32)
    subdiag = np.array([[0, -1, -1], [1, -2, -2]], dtype=np.float32)
    diagonals = (superdiag, maindiag, subdiag)
    rhs = np.array([[[1, 1], [1, 1], [1, 1]], [[2, 2], [2, 2], [2, 2]]], dtype=np.float32)
    diagonals_format = 'sequence'
    name = 'test8'
    input_dict = {'diagonals': diagonals, 'rhs': rhs, 'diagonals_format': diagonals_format, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    diagonals = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype=np.float64)
    rhs = np.array([[[[1, 1], [1, 1], [1, 1]]]], dtype=np.float64)
    diagonals_format = 'compact'
    name = 'test9'
    input_dict = {'diagonals': diagonals, 'rhs': rhs, 'diagonals_format': diagonals_format, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    superdiag = np.array([1.0+1.0j, 2.0+2.0j, 0.0], dtype=np.complex128)
    maindiag = np.array([4.0, 5.0, 6.0], dtype=np.complex128)
    subdiag = np.array([0.0, 8.0, 9.0+9.0j], dtype=np.complex128)
    diagonals = (superdiag, maindiag, subdiag)
    rhs = np.array([[1.0, 2.0+1.0j], [3.0+3.0j, 4.0], [5.0, 6.0]], dtype=np.complex128)

    diagonals_format = 'sequence'
    name = 'test10'
    input_dict = {'diagonals': diagonals, 'rhs': rhs, 'diagonals_format': diagonals_format, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.tridiagonal_matmul_2"] = tf_linalg_tridiagonal_matmul_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.tridiagonal_matmul_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.tridiagonal_matmul_2'.")

check_valid('tf.linalg.tridiagonal_matmul', generated_inputs['tf.linalg.tridiagonal_matmul_2'], lib="tf", suffix=2)
