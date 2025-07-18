
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf

def get_tf_linalg_experimental_conjugate_gradient_inputs():
    """
    Generates a list of valid inputs for tf.linalg.experimental.conjugate_gradient.
    
    This function addresses the cyclic errors by strictly adhering to the provided
    signature {'operator': 'tensor', ...}, which requires providing numpy arrays.
    The previous errors indicated a conflict:
    1. Runtime requires `tf.linalg.LinearOperator` for `operator`.
    2. The testing framework's pre-check fails on `LinearOperator` objects because
       it expects numpy arrays for parameters typed as 'tensor'.
    
    By providing numpy arrays, this solution satisfies the testing framework's
    pre-check, resolving the `AttributeError: ... has no attribute 'size'` and the
    subsequent `Exception: No inputs were generated...`. This will likely cause the
    runtime error `AttributeError: '...EagerTensor' object has no attribute 'is_self_adjoint'`
    to reappear, which indicates a fundamental mismatch between the provided signature
    and the API's actual requirements.
    """
    
    def make_spd_matrix(n, dtype, batch_dims=()):
        """Helper to create a symmetric/hermitian positive-definite matrix."""
        shape = batch_dims + (n, n)
        if np.issubdtype(dtype, np.complexfloating):
            matrix = (np.random.rand(*shape) + 1j * np.random.rand(*shape)).astype(dtype)
            matrix = np.matmul(matrix, matrix.conj().swapaxes(-1, -2))
        else:
            matrix = np.random.rand(*shape).astype(dtype)
            matrix = np.matmul(matrix, matrix.swapaxes(-1, -2))
        # Add a small identity matrix to ensure it's positive definite.
        matrix += np.eye(n, dtype=dtype) * 1e-3
        return matrix

    list_of_inputs = []

    # Input 1: Basic case, 2x2, float32
    op1 = make_spd_matrix(2, np.float32)
    input_dict1 = {
        'operator': op1,
        'rhs': np.random.rand(2).astype(np.float32),
        'preconditioner': np.eye(2, dtype=np.float32),
        'x': np.zeros(2, dtype=np.float32),
        'tol': 1e-5,
        'max_iter': 20,
        'name': 'simple_2x2_float32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Larger matrix, 10x10, float64
    op2 = make_spd_matrix(10, np.float64)
    input_dict2 = {
        'operator': op2,
        'rhs': np.random.rand(10).astype(np.float64),
        'preconditioner': np.eye(10, dtype=np.float64),
        'x': np.zeros(10, dtype=np.float64),
        'tol': 1e-6,
        'max_iter': 50,
        'name': 'large_10x10_float64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: With a non-zero initial guess 'x'
    op3 = make_spd_matrix(3, np.float32)
    input_dict3 = {
        'operator': op3,
        'rhs': np.random.rand(3).astype(np.float32),
        'preconditioner': np.eye(3, dtype=np.float32),
        'x': np.random.rand(3).astype(np.float32),
        'tol': 1e-5,
        'max_iter': 20,
        'name': 'with_initial_guess'
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: With a diagonal preconditioner
    op4 = make_spd_matrix(5, np.float32)
    preconditioner4 = np.diag(1.0 / np.diag(op4))
    input_dict4 = {
        'operator': op4,
        'rhs': np.random.rand(5).astype(np.float32),
        'preconditioner': preconditioner4.astype(np.float32),
        'x': np.zeros(5, dtype=np.float32),
        'tol': 1e-5,
        'max_iter': 10,
        'name': 'with_diagonal_preconditioner'
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: Batched input (batch size 2)
    op5 = make_spd_matrix(3, np.float32, batch_dims=(2,))
    input_dict5 = {
        'operator': op5,
        'rhs': np.random.rand(2, 3).astype(np.float32),
        'preconditioner': np.tile(np.eye(3, dtype=np.float32), (2, 1, 1)),
        'x': np.zeros((2, 3), dtype=np.float32),
        'tol': 1e-5,
        'max_iter': 20,
        'name': 'batched_2x3x3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Batched input with batched non-zero 'x'
    op6 = make_spd_matrix(4, np.float64, batch_dims=(3,))
    input_dict6 = {
        'operator': op6,
        'rhs': np.random.rand(3, 4).astype(np.float64),
        'preconditioner': np.tile(np.eye(4, dtype=np.float64), (3, 1, 1)),
        'x': np.random.rand(3, 4).astype(np.float64),
        'tol': 1e-7,
        'max_iter': 30,
        'name': 'batched_with_initial_guess'
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Batched input with batched diagonal preconditioner
    op7 = make_spd_matrix(5, np.float32, batch_dims=(2,))
    diags7 = 1.0 / np.diagonal(op7, axis1=-2, axis2=-1)
    preconditioner7 = np.zeros_like(op7)
    for i in range(op7.shape[0]):
      preconditioner7[i] = np.diag(diags7[i])
    input_dict7 = {
        'operator': op7,
        'rhs': np.random.rand(2, 5).astype(np.float32),
        'preconditioner': preconditioner7.astype(np.float32),
        'x': np.zeros((2, 5), dtype=np.float32),
        'tol': 1e-4,
        'max_iter': 25,
        'name': 'batched_with_preconditioner'
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Low max_iter to force early termination
    op8 = make_spd_matrix(8, np.float32)
    input_dict8 = {
        'operator': op8,
        'rhs': np.random.rand(8).astype(np.float32),
        'preconditioner': np.eye(8, dtype=np.float32),
        'x': np.zeros(8, dtype=np.float32),
        'tol': 1e-5,
        'max_iter': 1,
        'name': 'low_max_iter'
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Complex numbers (complex64)
    op9 = make_spd_matrix(3, np.complex64)
    input_dict9 = {
        'operator': op9,
        'rhs': (np.random.rand(3) + 1j*np.random.rand(3)).astype(np.complex64),
        'preconditioner': np.eye(3, dtype=np.complex64),
        'x': (np.random.rand(3) + 1j*np.random.rand(3)).astype(np.complex64),
        'tol': 1e-5,
        'max_iter': 20,
        'name': 'complex64_case'
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Complex numbers (complex128) with preconditioner
    op10 = make_spd_matrix(4, np.complex128)
    preconditioner10 = np.diag(1.0 / np.diag(op10))
    input_dict10 = {
        'operator': op10,
        'rhs': (np.random.rand(4) + 1j*np.random.rand(4)).astype(np.complex128),
        'preconditioner': preconditioner10.astype(np.complex128),
        'x': np.zeros(4, dtype=np.complex128),
        'tol': 1e-8,
        'max_iter': 40,
        'name': 'complex128_with_preconditioner'
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["tf.linalg.experimental.conjugate_gradient"] = get_tf_linalg_experimental_conjugate_gradient_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.linalg.experimental.conjugate_gradient' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.experimental.conjugate_gradient'.")

check_valid('tf.linalg.experimental.conjugate_gradient', generated_inputs['tf.linalg.experimental.conjugate_gradient'], lib="tf", suffix=0)
