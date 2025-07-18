
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_linalg_tridiagonal_matmul_inputs():
    """
    Generates a list of valid inputs for tf.linalg.tridiagonal_matmul.
    """
    list_of_inputs = []

    # Input 1: Basic 'compact' format, float32
    diagonals1 = np.array([
        [-1., -1., 0.],  # superdiagonal (last element ignored)
        [2., 2., 2.],   # main diagonal
        [0., -1., -1.]   # subdiagonal (first element ignored)
    ], dtype=np.float32)
    rhs1 = np.array([
        [1., 1.],
        [1., 1.],
        [1., 1.]
    ], dtype=np.float32)
    input_dict1 = {
        'diagonals': diagonals1,
        'rhs': rhs1,
        'diagonals_format': 'compact',
        'name': 'compact_basic_float32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Basic 'matrix' format, float64
    diagonals2 = np.array([
        [4., 1., 99., 99.], # Non-tridiagonal elements are ignored
        [2., 5., 1., 99.],
        [99., 3., 6., 1.],
        [99., 99., 4., 7.]
    ], dtype=np.float64)
    rhs2 = np.random.rand(4, 3).astype(np.float64)
    input_dict2 = {
        'diagonals': diagonals2,
        'rhs': rhs2,
        'diagonals_format': 'matrix',
        'name': 'matrix_basic_float64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 'compact' format with batching
    diagonals3 = np.random.rand(2, 3, 4).astype(np.float32)
    rhs3 = np.random.rand(2, 4, 2).astype(np.float32)
    input_dict3 = {
        'diagonals': diagonals3,
        'rhs': rhs3,
        'diagonals_format': 'compact',
        'name': 'compact_batched'
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 'matrix' format with batching
    diagonals4 = np.array([
        [[2, 1, 0], [1, 2, 1], [0, 1, 2]],
        [[5, -1, 0], [1, 5, -1], [0, 1, 5]]
    ], dtype=np.float64)
    rhs4 = np.ones((2, 3, 3), dtype=np.float64)
    input_dict4 = {
        'diagonals': diagonals4,
        'rhs': rhs4,
        'diagonals_format': 'matrix',
        'name': 'matrix_batched_float64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 'compact' format, wide rhs (N > M)
    diagonals5 = np.array([
        [1., 1., 0.],
        [3., 3., 3.],
        [0., 1., 1.]
    ], dtype=np.float32)
    rhs5 = np.random.rand(3, 5).astype(np.float32)
    input_dict5 = {
        'diagonals': diagonals5,
        'rhs': rhs5,
        'diagonals_format': 'compact',
        'name': 'compact_wide_rhs'
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 'matrix' format, tall rhs (M > N)
    diagonals6 = np.random.rand(5, 5).astype(np.float32) # will be treated as tridiagonal
    rhs6 = np.random.rand(5, 2).astype(np.float32)
    input_dict6 = {
        'diagonals': diagonals6,
        'rhs': rhs6,
        'diagonals_format': 'matrix',
        'name': 'matrix_tall_rhs'
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Negative values, 'compact' format
    diagonals7 = -np.abs(np.random.rand(3, 4)).astype(np.float64)
    rhs7 = -np.abs(np.random.rand(4, 4)).astype(np.float64)
    input_dict7 = {
        'diagonals': diagonals7,
        'rhs': rhs7,
        'diagonals_format': 'compact',
        'name': 'compact_negative_values'
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Complex numbers, 'matrix' format
    diagonals8_real = np.array([[2, 1, 0], [1, 2, 1], [0, 1, 2]])
    diagonals8_imag = np.array([[1, -1, 0], [1, 1, -1], [0, 1, 1]])
    diagonals8 = (diagonals8_real + 1j * diagonals8_imag).astype(np.complex64)
    rhs8 = (np.random.rand(3, 2) + 1j * np.random.rand(3, 2)).astype(np.complex64)
    input_dict8 = {
        'diagonals': diagonals8,
        'rhs': rhs8,
        'diagonals_format': 'matrix',
        'name': 'matrix_complex64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Higher-dimensional batching, 'compact' format
    diagonals9 = np.random.rand(2, 2, 3, 5).astype(np.float32)
    rhs9 = np.random.rand(2, 2, 5, 3).astype(np.float32)
    input_dict9 = {
        'diagonals': diagonals9,
        'rhs': rhs9,
        'diagonals_format': 'compact',
        'name': 'compact_high_dim_batch'
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Larger M, 'compact' format
    M = 10
    N = 4
    diagonals10 = np.random.rand(3, M).astype(np.float64)
    rhs10 = np.random.rand(M, N).astype(np.float64)
    input_dict10 = {
        'diagonals': diagonals10,
        'rhs': rhs10,
        'diagonals_format': 'compact',
        'name': 'compact_large_M'
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    # Input 11: M=1 case, 'compact'
    diagonals11 = np.array([
        [0.], # super (ignored)
        [5.], # main
        [0.]  # sub (ignored)
    ], dtype=np.float32)
    rhs11 = np.array([[10., 11., 12.]], dtype=np.float32)
    input_dict11 = {
        'diagonals': diagonals11,
        'rhs': rhs11,
        'diagonals_format': 'compact',
        'name': 'compact_M1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict11))
    
    # Input 12: M=2 case, 'matrix'
    diagonals12 = np.array([
        [10., 2.],
        [3., 11.]
    ], dtype=np.float64)
    rhs12 = np.array([
        [1., 0.],
        [0., 1.]
    ], dtype=np.float64)
    input_dict12 = {
        'diagonals': diagonals12,
        'rhs': rhs12,
        'diagonals_format': 'matrix',
        'name': 'matrix_M2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict12))

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

check_valid('tf.linalg.tridiagonal_matmul', generated_inputs['tf.linalg.tridiagonal_matmul'], lib="tf", suffix=0)
