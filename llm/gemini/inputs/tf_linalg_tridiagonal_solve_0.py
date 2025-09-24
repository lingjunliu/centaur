
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_linalg_tridiagonal_solve_inputs():
    """
    Generates a list of valid inputs for tf.linalg.tridiagonal_solve.
    """
    list_of_inputs = []

    # Input 1: Basic 'compact' format, float32, single RHS
    diagonals1 = np.array([
        [1., 2., 3., 0.],       # superdiagonal (last element ignored)
        [10., 11., 12., 13.],    # diagonal
        [0., 4., 5., 6.]        # subdiagonal (first element ignored)
    ], dtype=np.float32)
    rhs1 = np.array([8., 15., 22., 29.], dtype=np.float32)
    input_dict1 = {
        'diagonals': diagonals1,
        'rhs': rhs1,
        'diagonals_format': 'compact',
        'transpose_rhs': False,
        'conjugate_rhs': False,
        'name': 'compact_float32_single_rhs',
        'partial_pivoting': True,
        'perturb_singular': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Basic 'matrix' format, float64, single RHS
    diagonals2 = np.array([
        [4., 1., 0., 0.],
        [-1., 4., 1., 0.],
        [0., -1., 4., 1.],
        [0., 0., -1., 4.]
    ], dtype=np.float64)
    rhs2 = np.array([1., 2., 3., 4.], dtype=np.float64)
    input_dict2 = {
        'diagonals': diagonals2,
        'rhs': rhs2,
        'diagonals_format': 'matrix',
        'transpose_rhs': False,
        'conjugate_rhs': False,
        'name': 'matrix_float64_single_rhs',
        'partial_pivoting': True,
        'perturb_singular': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 'compact' format with batch dimension (B=2, M=3)
    diagonals3 = np.array([
        [[1., 2., 0.], [10., 11., 12.], [0., 3., 4.]],
        [[5., 6., 0.], [20., 21., 22.], [0., 7., 8.]]
    ], dtype=np.float32)
    rhs3 = np.random.rand(2, 3).astype(np.float32)
    input_dict3 = {
        'diagonals': diagonals3,
        'rhs': rhs3,
        'diagonals_format': 'compact',
        'transpose_rhs': False,
        'conjugate_rhs': False,
        'name': 'compact_batched',
        'partial_pivoting': True,
        'perturb_singular': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 'matrix' format with multiple RHS (K=2) and no pivoting
    # Using a symmetric positive definite matrix, so pivoting is not necessary.
    diagonals4 = np.array([
        [10., -1., 0.],
        [-1., 10., -1.],
        [0., -1., 10.]
    ], dtype=np.float64)
    rhs4 = np.random.rand(3, 2).astype(np.float64)
    input_dict4 = {
        'diagonals': diagonals4,
        'rhs': rhs4,
        'diagonals_format': 'matrix',
        'transpose_rhs': False,
        'conjugate_rhs': False,
        'name': 'matrix_multiple_rhs_no_pivoting',
        'partial_pivoting': False,
        'perturb_singular': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 'compact' format (converted from sequence), batched multiple RHS and transpose_rhs=True
    super5 = np.random.rand(2, 4).astype(np.float32)
    main5 = np.random.rand(2, 5).astype(np.float32) * 10
    sub5 = np.random.rand(2, 4).astype(np.float32)
    padded_super5 = np.pad(super5, ((0, 0), (0, 1)), 'constant')
    padded_sub5 = np.pad(sub5, ((0, 0), (1, 0)), 'constant')
    diagonals5 = np.stack([padded_super5, main5, padded_sub5], axis=1) # shape (2, 3, 5)
    rhs5 = np.random.rand(2, 3, 5).astype(np.float32) # shape [B, K, M]
    input_dict5 = {
        'diagonals': diagonals5,
        'rhs': rhs5,
        'diagonals_format': 'compact',
        'transpose_rhs': True,
        'conjugate_rhs': False,
        'name': 'compact_batched_multi_rhs_transposed',
        'partial_pivoting': True,
        'perturb_singular': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: Complex type (complex64) with conjugate_rhs=True, 'compact' format
    diagonals6 = np.array([
        [1.+1.j, 2.+2.j, 0.+0.j],
        [10.+1j, 11.+1j, 12.+1j],
        [0.+0.j, 4.-3.j, 5.-4.j]
    ], dtype=np.complex64)
    rhs6 = np.array([1.+2.j, 3.+4.j, 5.+6.j], dtype=np.complex64)
    input_dict6 = {
        'diagonals': diagonals6,
        'rhs': rhs6,
        'diagonals_format': 'compact',
        'transpose_rhs': False,
        'conjugate_rhs': True,
        'name': 'complex64_conjugate_rhs',
        'partial_pivoting': True,
        'perturb_singular': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Complex type (complex128) with multiple RHS and transpose + conjugate, 'matrix' format
    mat7 = np.zeros((4, 4), dtype=np.complex128)
    mat7.flat[::5] = np.random.rand(4) + 1j * np.random.rand(4) # main diag
    mat7.flat[1::5] = np.random.rand(3) + 1j * np.random.rand(3) # super diag
    mat7.flat[4::5] = np.random.rand(3) + 1j * np.random.rand(3) # sub diag
    rhs7 = (np.random.rand(2, 4) + 1j * np.random.rand(2, 4)).astype(np.complex128) # shape [K, M]
    input_dict7 = {
        'diagonals': mat7,
        'rhs': rhs7,
        'diagonals_format': 'matrix',
        'transpose_rhs': True,
        'conjugate_rhs': True,
        'name': 'complex128_trans_conj_rhs',
        'partial_pivoting': True,
        'perturb_singular': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: 'compact' format (converted from sequence) where diagonals have length M
    super8 = np.array([1., 2., 3., 99.], dtype=np.float32)
    main8 = np.array([10., 11., 12., 13.], dtype=np.float32)
    sub8 = np.array([99., 4., 5., 6.], dtype=np.float32)
    diagonals8 = np.stack([super8, main8, sub8])
    rhs8 = np.array([1., 2., 3., 4.], dtype=np.float32)
    input_dict8 = {
        'diagonals': diagonals8,
        'rhs': rhs8,
        'diagonals_format': 'compact',
        'transpose_rhs': False,
        'conjugate_rhs': False,
        'name': 'compact_N_equals_M',
        'partial_pivoting': True,
        'perturb_singular': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Batched matrix format with float64
    diagonals9 = np.array([
        [[4., 1., 0.], [-1., 4., 1.], [0., -1., 4.]],
        [[5., 2., 0.], [-2., 5., 2.], [0., -2., 5.]]
    ], dtype=np.float64)
    rhs9 = np.random.rand(2, 3).astype(np.float64)
    input_dict9 = {
        'diagonals': diagonals9,
        'rhs': rhs9,
        'diagonals_format': 'matrix',
        'transpose_rhs': False,
        'conjugate_rhs': False,
        'name': 'matrix_batched_float64',
        'partial_pivoting': True,
        'perturb_singular': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Perturb singular matrix
    # This matrix is singular: det = 1*(2*1 - 1*1) - 1*(1*1) = 0
    diagonals10 = np.array([
        [1., 1., 99.],  # super: [1, 1], last ignored
        [1., 2., 1.],   # diag: [1, 2, 1]
        [99., 1., 1.]   # sub: [1, 1], first ignored
    ], dtype=np.float32)
    rhs10 = np.array([1., 2., 1.], dtype=np.float32)
    input_dict10 = {
        'diagonals': diagonals10,
        'rhs': rhs10,
        'diagonals_format': 'compact',
        'transpose_rhs': False,
        'conjugate_rhs': False,
        'name': 'perturb_singular_example',
        'partial_pivoting': True,
        'perturb_singular': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

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

check_valid('tf.linalg.tridiagonal_solve', generated_inputs['tf.linalg.tridiagonal_solve'], lib="tf", suffix=0)
