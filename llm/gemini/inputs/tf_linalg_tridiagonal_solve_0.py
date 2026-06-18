
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_tridiagonal_solve_inputs():
    list_of_inputs = []
    
    # 1. Compact format, float32, no batch, M=4
    diagonals = np.array([
        [1.0, 1.0, 1.0, 0.0],
        [4.0, 4.0, 4.0, 4.0],
        [0.0, 1.0, 1.0, 1.0]
    ], dtype=np.float32)
    rhs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    input_dict = {
        'diagonals': diagonals,
        'rhs': rhs,
        'diagonals_format': 'compact',
        'transpose_rhs': False,
        'conjugate_rhs': False,
        'name': 'solve_1',
        'partial_pivoting': True,
        'perturb_singular': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2. Compact format, float64, batch [2], M=3, multiple RHS (K=2)
    diagonals = np.zeros((2, 3, 3), dtype=np.float64)
    diagonals[:, 0, :] = 1.0
    diagonals[:, 1, :] = 4.0
    diagonals[:, 2, :] = 1.0
    rhs = np.ones((2, 3, 2), dtype=np.float64)
    input_dict = {
        'diagonals': diagonals,
        'rhs': rhs,
        'diagonals_format': 'compact',
        'transpose_rhs': False,
        'conjugate_rhs': False,
        'name': 'solve_2',
        'partial_pivoting': True,
        'perturb_singular': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3. Matrix format, float32, no batch, M=3
    diagonals = np.array([
        [4.0, 1.0, 0.0],
        [1.0, 4.0, 1.0],
        [0.0, 1.0, 4.0]
    ], dtype=np.float32)
    rhs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {
        'diagonals': diagonals,
        'rhs': rhs,
        'diagonals_format': 'matrix',
        'transpose_rhs': False,
        'conjugate_rhs': False,
        'name': 'solve_3',
        'partial_pivoting': False,
        'perturb_singular': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 4. Compact format, complex64, M=3, transpose_rhs=True, K=2 (rhs shape [K, M])
    diagonals = np.zeros((3, 3), dtype=np.complex64)
    diagonals[0, :] = 1.0 + 0j
    diagonals[1, :] = 4.0 + 0j
    diagonals[2, :] = 1.0 + 0j
    rhs = np.array([[1.0 + 1j, 2.0 + 2j, 3.0 + 3j], [4.0, 5.0, 6.0]], dtype=np.complex64)
    input_dict = {
        'diagonals': diagonals,
        'rhs': rhs,
        'diagonals_format': 'compact',
        'transpose_rhs': True,
        'conjugate_rhs': True,
        'name': 'solve_4',
        'partial_pivoting': True,
        'perturb_singular': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 5. Matrix format, batch shape [1, 2], float32, M=3
    diagonals = np.zeros((1, 2, 3, 3), dtype=np.float32)
    for i in range(3):
        diagonals[..., i, i] = 4.0
        if i > 0:
            diagonals[..., i, i-1] = 1.0
        if i < 2:
            diagonals[..., i, i+1] = 1.0
    rhs = np.ones((1, 2, 3), dtype=np.float32)
    input_dict = {
        'diagonals': diagonals,
        'rhs': rhs,
        'diagonals_format': 'matrix',
        'transpose_rhs': False,
        'conjugate_rhs': False,
        'name': 'solve_5',
        'partial_pivoting': True,
        'perturb_singular': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 6. Compact format, float32, batch shape [3], M=5, partial_pivoting=False
    diagonals = np.zeros((3, 3, 5), dtype=np.float32)
    diagonals[:, 0, :] = 1.0
    diagonals[:, 1, :] = 5.0
    diagonals[:, 2, :] = 1.0
    rhs = np.ones((3, 5, 2), dtype=np.float32)
    input_dict = {
        'diagonals': diagonals,
        'rhs': rhs,
        'diagonals_format': 'compact',
        'transpose_rhs': False,
        'conjugate_rhs': False,
        'name': 'solve_6',
        'partial_pivoting': False,
        'perturb_singular': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 7. Compact format, complex128, M=4
    diagonals = np.zeros((3, 4), dtype=np.complex128)
    diagonals[0, :] = 1.0 + 0j
    diagonals[1, :] = 4.0 + 0j
    diagonals[2, :] = 1.0 + 0j
    rhs = np.array([1.0 + 1j, 2.0 + 2j, 3.0 + 3j, 4.0 + 4j], dtype=np.complex128)
    input_dict = {
        'diagonals': diagonals,
        'rhs': rhs,
        'diagonals_format': 'compact',
        'transpose_rhs': False,
        'conjugate_rhs': True,
        'name': 'solve_7',
        'partial_pivoting': True,
        'perturb_singular': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 8. Matrix format, float64, batch [2], M=3, RHS [2, 3]
    diagonals = np.zeros((2, 3, 3), dtype=np.float64)
    for b in range(2):
        for i in range(3):
            diagonals[b, i, i] = 4.0
            if i > 0:
                diagonals[b, i, i-1] = 1.0
            if i < 2:
                diagonals[b, i, i+1] = 1.0
    rhs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    input_dict = {
        'diagonals': diagonals,
        'rhs': rhs,
        'diagonals_format': 'matrix',
        'transpose_rhs': False,
        'conjugate_rhs': False,
        'name': 'solve_8',
        'partial_pivoting': True,
        'perturb_singular': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 9. Compact format, float32, M=2
    diagonals = np.array([
        [1.0, 0.0],
        [4.0, 4.0],
        [0.0, 1.0]
    ], dtype=np.float32)
    rhs = np.array([2.0, 3.0], dtype=np.float32)
    input_dict = {
        'diagonals': diagonals,
        'rhs': rhs,
        'diagonals_format': 'compact',
        'transpose_rhs': False,
        'conjugate_rhs': False,
        'name': 'solve_9',
        'partial_pivoting': False,
        'perturb_singular': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 10. Matrix format, float32, M=4, transpose_rhs=True, K=2 (rhs shape [2, 4])
    diagonals = np.zeros((4, 4), dtype=np.float32)
    for i in range(4):
        diagonals[i, i] = 4.0
        if i > 0:
            diagonals[i, i-1] = 1.0
        if i < 3:
            diagonals[i, i+1] = 1.0
    rhs = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
    input_dict = {
        'diagonals': diagonals,
        'rhs': rhs,
        'diagonals_format': 'matrix',
        'transpose_rhs': True,
        'conjugate_rhs': False,
        'name': 'solve_10',
        'partial_pivoting': True,
        'perturb_singular': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.linalg.tridiagonal_solve"] = tf_linalg_tridiagonal_solve_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.linalg.tridiagonal_solve' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.tridiagonal_solve'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.linalg.tridiagonal_solve', generated_inputs['tf.linalg.tridiagonal_solve'], lib="tf", suffix=0)
