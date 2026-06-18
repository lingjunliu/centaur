
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_tridiagonal_matmul_inputs():
    list_of_inputs = []

    # Input 1: Basic float32, no batch
    diagonals = np.array([
        [1.0, 2.0, 0.0],  # superdiag
        [3.0, 4.0, 5.0],  # maindiag
        [0.0, 6.0, 7.0]   # subdiag
    ], dtype=np.float32)
    rhs = np.array([
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0]
    ], dtype=np.float32)
    input_dict = {
        'diagonals': diagonals,
        'rhs': rhs,
        'diagonals_format': 'compact',
        'name': 'matmul_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, with negative values
    diagonals = np.array([
        [-1.0, -2.0, 0.0, 0.0],
        [2.0, 2.0, -2.0, 2.0],
        [0.0, -1.0, -1.0, -3.0]
    ], dtype=np.float64)
    rhs = np.array([
        [1.0],
        [-1.0],
        [2.0],
        [-2.0]
    ], dtype=np.float64)
    input_dict = {
        'diagonals': diagonals,
        'rhs': rhs,
        'diagonals_format': 'compact',
        'name': 'matmul_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Batch dimension 1, float32
    diagonals = np.random.randn(2, 3, 5).astype(np.float32)
    rhs = np.random.randn(2, 5, 3).astype(np.float32)
    input_dict = {
        'diagonals': diagonals,
        'rhs': rhs,
        'diagonals_format': 'compact',
        'name': 'matmul_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Complex64
    diagonals = (np.random.randn(3, 4) + 1j * np.random.randn(3, 4)).astype(np.complex64)
    rhs = (np.random.randn(4, 2) + 1j * np.random.randn(4, 2)).astype(np.complex64)
    input_dict = {
        'diagonals': diagonals,
        'rhs': rhs,
        'diagonals_format': 'compact',
        'name': 'matmul_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex128, batched
    diagonals = (np.random.randn(2, 3, 3) + 1j * np.random.randn(2, 3, 3)).astype(np.complex128)
    rhs = (np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)).astype(np.complex128)
    input_dict = {
        'diagonals': diagonals,
        'rhs': rhs,
        'diagonals_format': 'compact',
        'name': 'matmul_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large M and N, float64
    diagonals = np.random.randn(3, 100).astype(np.float64)
    rhs = np.random.randn(100, 50).astype(np.float64)
    input_dict = {
        'diagonals': diagonals,
        'rhs': rhs,
        'diagonals_format': 'compact',
        'name': 'matmul_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Batch dimensions 2, float32
    diagonals = np.random.randn(3, 2, 3, 6).astype(np.float32)
    rhs = np.random.randn(3, 2, 6, 2).astype(np.float32)
    input_dict = {
        'diagonals': diagonals,
        'rhs': rhs,
        'diagonals_format': 'compact',
        'name': 'matmul_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small size M=2, N=1
    diagonals = np.array([
        [0.5, 0.0],
        [1.5, 2.5],
        [0.0, 3.5]
    ], dtype=np.float32)
    rhs = np.array([
        [1.0],
        [2.0]
    ], dtype=np.float32)
    input_dict = {
        'diagonals': diagonals,
        'rhs': rhs,
        'diagonals_format': 'compact',
        'name': 'matmul_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Zero arrays
    diagonals = np.zeros((3, 10), dtype=np.float32)
    rhs = np.zeros((10, 10), dtype=np.float32)
    input_dict = {
        'diagonals': diagonals,
        'rhs': rhs,
        'diagonals_format': 'compact',
        'name': 'matmul_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Single batch large size, float32
    diagonals = np.random.randn(5, 3, 20).astype(np.float32)
    rhs = np.random.randn(5, 20, 10).astype(np.float32)
    input_dict = {
        'diagonals': diagonals,
        'rhs': rhs,
        'diagonals_format': 'compact',
        'name': 'matmul_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.linalg.tridiagonal_matmul"] = tf_linalg_tridiagonal_matmul_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.linalg.tridiagonal_matmul' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.tridiagonal_matmul'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.linalg.tridiagonal_matmul', generated_inputs['tf.linalg.tridiagonal_matmul'], lib="tf", suffix=0)
