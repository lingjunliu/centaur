
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_tf_linalg_tridiagonal_matmul_inputs():
    """
    Generates a list of valid inputs for the tf.linalg.tridiagonal_matmul function.
    """
    list_of_inputs = []

    # All inputs will use 'compact' format to ensure 'diagonals' is a single tensor.

    # Input 1: Basic 'compact' format, float32
    diagonals_1 = np.array([
        [1, 2, 3, 0],   # super
        [4, 5, 6, 7],   # main
        [0, 8, 9, 10]   # sub
    ], dtype=np.float32)
    rhs_1 = np.random.rand(4, 2).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        'diagonals': diagonals_1,
        'rhs': rhs_1,
        'diagonals_format': 'compact',
        'name': 'compact_float32'
    }))

    # Input 2: Basic 'compact' format, float64 with negative values
    diagonals_2 = np.array([
        [-1, -2, -3, -4, 0],  # superdiagonal (last element ignored)
        [10, 20, 30, 40, 50], # main diagonal
        [0, -5, -6, -7, -8]   # subdiagonal (first element ignored)
    ], dtype=np.float64)
    rhs_2 = np.random.rand(5, 3).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({
        'diagonals': diagonals_2,
        'rhs': rhs_2,
        'diagonals_format': 'compact',
        'name': 'compact_float64_neg'
    }))

    # Input 3: 'compact' format with one batch dimension, float32
    diagonals_3 = np.random.rand(2, 3, 4).astype(np.float32)
    rhs_3 = np.random.rand(2, 4, 3).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        'diagonals': diagonals_3,
        'rhs': rhs_3,
        'diagonals_format': 'compact',
        'name': 'compact_batch_float32'
    }))

    # Input 4: 'compact' format with two batch dimensions, float64
    diagonals_4 = np.random.rand(2, 2, 3, 5).astype(np.float64)
    rhs_4 = np.random.rand(2, 2, 5, 1).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({
        'diagonals': diagonals_4,
        'rhs': rhs_4,
        'diagonals_format': 'compact',
        'name': 'compact_batch2d_float64'
    }))

    # Input 5: 'compact' format, complex64
    diagonals_5 = (np.random.rand(3, 3) + 1j * np.random.rand(3, 3)).astype(np.complex64)
    rhs_5 = (np.random.rand(3, 3) + 1j * np.random.rand(3, 3)).astype(np.complex64)
    list_of_inputs.append(copy.deepcopy({
        'diagonals': diagonals_5,
        'rhs': rhs_5,
        'diagonals_format': 'compact',
        'name': 'compact_complex64'
    }))

    # Input 6: 'compact' format, complex128
    diagonals_6 = (np.random.rand(3, 6) + 1j * np.random.rand(3, 6)).astype(np.complex128)
    rhs_6 = (np.random.rand(6, 2) + 1j * np.random.rand(6, 2)).astype(np.complex128)
    list_of_inputs.append(copy.deepcopy({
        'diagonals': diagonals_6,
        'rhs': rhs_6,
        'diagonals_format': 'compact',
        'name': 'compact_complex128'
    }))

    # Input 7: 'compact' with batch and complex numbers, complex64
    diagonals_7 = (np.random.rand(3, 3, 5) + 1j * np.random.rand(3, 3, 5)).astype(np.complex64)
    rhs_7 = (np.random.rand(3, 5, 5) + 1j * np.random.rand(3, 5, 5)).astype(np.complex64)
    list_of_inputs.append(copy.deepcopy({
        'diagonals': diagonals_7,
        'rhs': rhs_7,
        'diagonals_format': 'compact',
        'name': 'compact_batch_complex64'
    }))

    # Input 8: 'compact' with batch and complex numbers, complex128
    diagonals_8 = (np.random.rand(2, 3, 4) + 1j * np.random.rand(2, 3, 4)).astype(np.complex128)
    rhs_8 = (np.random.rand(2, 4, 3) + 1j * np.random.rand(2, 4, 3)).astype(np.complex128)
    list_of_inputs.append(copy.deepcopy({
        'diagonals': diagonals_8,
        'rhs': rhs_8,
        'diagonals_format': 'compact',
        'name': 'compact_batch_complex128'
    }))

    # Input 9: 'compact' format, larger M
    diagonals_9 = np.random.randn(3, 10).astype(np.float32)
    rhs_9 = np.random.randn(10, 5).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        'diagonals': diagonals_9,
        'rhs': rhs_9,
        'diagonals_format': 'compact',
        'name': 'large_M_compact'
    }))

    # Input 10: Minimal case, M=2, 'compact'
    diagonals_10 = np.array([[-1, 0], [2, 2], [0, -1]], dtype=np.float32)
    rhs_10 = np.eye(2, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({
        'diagonals': diagonals_10,
        'rhs': rhs_10,
        'diagonals_format': 'compact',
        'name': 'minimal_M_compact'
    }))

    # Input 11: Minimal case, M=1, 'compact'
    diagonals_11 = np.array([[0], [5], [0]], dtype=np.float64)
    rhs_11 = np.array([[10]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({
        'diagonals': diagonals_11,
        'rhs': rhs_11,
        'diagonals_format': 'compact',
        'name': 'M1_compact'
    }))

    # Input 12: 'compact' with N=1 (rhs is a vector)
    diagonals_12 = np.random.rand(3, 5).astype(np.float32)
    rhs_12 = np.random.rand(5, 1).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        'diagonals': diagonals_12,
        'rhs': rhs_12,
        'diagonals_format': 'compact',
        'name': 'N1_compact'
    }))
    
    return list_of_inputs

generated_inputs["tf.linalg.tridiagonal_matmul"] = get_tf_linalg_tridiagonal_matmul_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.linalg.tridiagonal_matmul' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.tridiagonal_matmul'.")

check_valid('tf.linalg.tridiagonal_matmul', generated_inputs['tf.linalg.tridiagonal_matmul'], lib="tf", suffix=0)
