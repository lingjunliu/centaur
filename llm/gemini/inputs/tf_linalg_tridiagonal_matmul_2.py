
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_tridiagonal_matmul_inputs():
    list_of_inputs = []

    # Input 1: Float32, no batch, M=3, N=2
    superdiag = np.array([1.0, 2.0, 0.0], dtype=np.float32)
    maindiag = np.array([3.0, 4.0, 5.0], dtype=np.float32)
    subdiag = np.array([0.0, 6.0, 7.0], dtype=np.float32)
    rhs = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    input_dict = {
        'diagonals': (superdiag, maindiag, subdiag),
        'rhs': rhs,
        'diagonals_format': 'sequence',
        'name': 'matmul_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float64, batch [2], M=4, N=1
    superdiag = np.random.randn(2, 4).astype(np.float64)
    maindiag = np.random.randn(2, 4).astype(np.float64)
    subdiag = np.random.randn(2, 4).astype(np.float64)
    rhs = np.random.randn(2, 4, 1).astype(np.float64)
    input_dict = {
        'diagonals': (superdiag, maindiag, subdiag),
        'rhs': rhs,
        'diagonals_format': 'sequence',
        'name': 'matmul_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Complex64, batch [3, 2], M=2, N=3
    superdiag = (np.random.randn(3, 2, 2) + 1j * np.random.randn(3, 2, 2)).astype(np.complex64)
    maindiag = (np.random.randn(3, 2, 2) + 1j * np.random.randn(3, 2, 2)).astype(np.complex64)
    subdiag = (np.random.randn(3, 2, 2) + 1j * np.random.randn(3, 2, 2)).astype(np.complex64)
    rhs = (np.random.randn(3, 2, 2, 3) + 1j * np.random.randn(3, 2, 2, 3)).astype(np.complex64)
    input_dict = {
        'diagonals': (superdiag, maindiag, subdiag),
        'rhs': rhs,
        'diagonals_format': 'sequence',
        'name': 'matmul_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Complex128, no batch, M=5, N=5
    superdiag = np.zeros(5, dtype=np.complex128)
    maindiag = np.ones(5, dtype=np.complex128)
    subdiag = np.zeros(5, dtype=np.complex128)
    rhs = np.eye(5, dtype=np.complex128)
    input_dict = {
        'diagonals': (superdiag, maindiag, subdiag),
        'rhs': rhs,
        'diagonals_format': 'sequence',
        'name': 'matmul_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Float32, batch [1], M=1, N=1
    superdiag = np.array([[0.0]], dtype=np.float32)
    maindiag = np.array([[5.0]], dtype=np.float32)
    subdiag = np.array([[0.0]], dtype=np.float32)
    rhs = np.array([[[2.0]]], dtype=np.float32)
    input_dict = {
        'diagonals': (superdiag, maindiag, subdiag),
        'rhs': rhs,
        'diagonals_format': 'sequence',
        'name': 'matmul_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Float64, negative values, M=3, N=3
    superdiag = np.array([-1.5, -2.5, 0.0], dtype=np.float64)
    maindiag = np.array([-3.5, -4.5, -5.5], dtype=np.float64)
    subdiag = np.array([0.0, -6.5, -7.5], dtype=np.float64)
    rhs = np.array([[-1.0, 0.0, 1.0], [2.0, -2.0, 2.0], [-3.0, 3.0, -3.0]], dtype=np.float64)
    input_dict = {
        'diagonals': (superdiag, maindiag, subdiag),
        'rhs': rhs,
        'diagonals_format': 'sequence',
        'name': 'matmul_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Float32, batch [2], M=10, N=5
    superdiag = np.random.uniform(-1, 1, (2, 10)).astype(np.float32)
    maindiag = np.random.uniform(-1, 1, (2, 10)).astype(np.float32)
    subdiag = np.random.uniform(-1, 1, (2, 10)).astype(np.float32)
    rhs = np.random.uniform(-1, 1, (2, 10, 5)).astype(np.float32)
    input_dict = {
        'diagonals': (superdiag, maindiag, subdiag),
        'rhs': rhs,
        'diagonals_format': 'sequence',
        'name': 'matmul_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Complex64, zeros, M=4, N=2
    superdiag = np.zeros((4,), dtype=np.complex64)
    maindiag = np.zeros((4,), dtype=np.complex64)
    subdiag = np.zeros((4,), dtype=np.complex64)
    rhs = np.zeros((4, 2), dtype=np.complex64)
    input_dict = {
        'diagonals': (superdiag, maindiag, subdiag),
        'rhs': rhs,
        'diagonals_format': 'sequence',
        'name': 'matmul_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Float64, high-dimensional batch [2, 2, 2], M=2, N=2
    superdiag = np.random.randn(2, 2, 2, 2).astype(np.float64)
    maindiag = np.random.randn(2, 2, 2, 2).astype(np.float64)
    subdiag = np.random.randn(2, 2, 2, 2).astype(np.float64)
    rhs = np.random.randn(2, 2, 2, 2, 2).astype(np.float64)
    input_dict = {
        'diagonals': (superdiag, maindiag, subdiag),
        'rhs': rhs,
        'diagonals_format': 'sequence',
        'name': 'matmul_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Float32, large identity-like values, M=3, N=3
    superdiag = np.zeros(3, dtype=np.float32)
    maindiag = np.ones(3, dtype=np.float32) * 10.0
    subdiag = np.zeros(3, dtype=np.float32)
    rhs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    input_dict = {
        'diagonals': (superdiag, maindiag, subdiag),
        'rhs': rhs,
        'diagonals_format': 'sequence',
        'name': 'matmul_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.linalg.tridiagonal_matmul_2"] = tf_linalg_tridiagonal_matmul_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.linalg.tridiagonal_matmul_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.tridiagonal_matmul_2'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.linalg.tridiagonal_matmul', generated_inputs['tf.linalg.tridiagonal_matmul_2'], lib="tf", suffix=2)
