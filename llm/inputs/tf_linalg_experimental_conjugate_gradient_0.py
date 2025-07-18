
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_conjugate_gradient_inputs():
    """
    Generates a list of valid inputs for the tf.linalg.experimental.conjugate_gradient function.
    """
    list_of_inputs = []

    def create_spd_matrix(shape, dtype):
        """Creates a symmetric positive-definite numpy matrix."""
        # Ensure the dtype for random values is float
        float_dtype = dtype.real if np.issubdtype(dtype, np.complexfloating) else dtype
        matrix = np.random.rand(*shape).astype(float_dtype)
        if np.issubdtype(dtype, np.complexfloating):
            matrix = matrix + 1j * np.random.rand(*shape).astype(float_dtype)

        if len(shape) > 2:
            transpose_axes = list(range(len(shape)))
            transpose_axes[-1], transpose_axes[-2] = transpose_axes[-2], transpose_axes[-1]
            psd_matrix = matrix @ np.transpose(matrix, axes=transpose_axes).conj()
            identity = np.eye(shape[-1], dtype=float_dtype)
            for _ in range(len(shape) - 2):
                identity = np.expand_dims(identity, 0)
            return (psd_matrix + 1e-4 * identity).astype(dtype)
        else:
            psd_matrix = matrix @ matrix.T.conj()
            return (psd_matrix + 1e-4 * np.eye(shape[0], dtype=float_dtype)).astype(dtype)

    # All tensor-like inputs are provided as numpy arrays to satisfy the test harness,
    # which expects array-like objects with a .size attribute. The harness is responsible
    # for converting these to the final required types for the API call.

    # Input 1: Basic 2x2 float32 case
    input_dict_1 = {
        'operator': create_spd_matrix((2, 2), np.float32),
        'rhs': np.array([1.0, 2.0], dtype=np.float32),
        'preconditioner': None,
        'x': None,
        'tol': 1e-5,
        'max_iter': 20,
        'name': 'basic_float32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Basic 3x3 float64 case
    input_dict_2 = {
        'operator': create_spd_matrix((3, 3), np.float64),
        'rhs': np.array([1.0, 2.0, 3.0], dtype=np.float64),
        'preconditioner': None,
        'x': np.zeros(3, dtype=np.float64),
        'tol': 1e-6,
        'max_iter': 30,
        'name': 'basic_float64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Non-zero initial guess
    input_dict_3 = {
        'operator': create_spd_matrix((4, 4), np.float32),
        'rhs': np.random.rand(4).astype(np.float32),
        'preconditioner': None,
        'x': np.random.rand(4).astype(np.float32),
        'tol': 1e-4,
        'max_iter': 10,
        'name': 'nonzero_initial_guess'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Diagonal preconditioner
    op_matrix_4 = create_spd_matrix((5, 5), np.float32)
    diag_precond_4 = np.diag(1.0 / np.diag(op_matrix_4))
    input_dict_4 = {
        'operator': op_matrix_4,
        'rhs': np.random.rand(5).astype(np.float32),
        'preconditioner': diag_precond_4,
        'x': np.zeros(5, dtype=np.float32),
        'tol': 1e-5,
        'max_iter': 25,
        'name': 'diagonal_preconditioner'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Batched input (batch_size=2)
    input_dict_5 = {
        'operator': create_spd_matrix((2, 3, 3), np.float32),
        'rhs': np.random.rand(2, 3).astype(np.float32),
        'preconditioner': None,
        'x': np.zeros([2, 3], dtype=np.float32),
        'tol': 1e-5,
        'max_iter': 20,
        'name': 'batched_float32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Batched input (batch_size=3) with float64
    input_dict_6 = {
        'operator': create_spd_matrix((3, 4, 4), np.float64),
        'rhs': np.random.rand(3, 4).astype(np.float64),
        'preconditioner': None,
        'x': None,
        'tol': 1e-7,
        'max_iter': 50,
        'name': 'batched_float64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Complex64 type
    input_dict_7 = {
        'operator': create_spd_matrix((3, 3), np.complex64),
        'rhs': (np.random.rand(3) + 1j * np.random.rand(3)).astype(np.complex64),
        'preconditioner': None,
        'x': np.zeros(3, dtype=np.complex64),
        'tol': 1e-5,
        'max_iter': 30,
        'name': 'complex64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Complex128 type with diagonal preconditioner
    op_matrix_8 = create_spd_matrix((4, 4), np.complex128)
    diag_precond_8 = np.diag(1.0 / np.diag(op_matrix_8).real).astype(np.complex128)
    input_dict_8 = {
        'operator': op_matrix_8,
        'rhs': (np.random.rand(4) + 1j * np.random.rand(4)).astype(np.complex128),
        'preconditioner': diag_precond_8,
        'x': np.zeros(4, dtype=np.complex128),
        'tol': 1e-8,
        'max_iter': 50,
        'name': 'complex128_preconditioned'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Large matrix with high max_iter
    input_dict_9 = {
        'operator': create_spd_matrix((50, 50), np.float32),
        'rhs': np.random.rand(50).astype(np.float32),
        'preconditioner': None,
        'x': np.zeros(50, dtype=np.float32),
        'tol': 1e-5,
        'max_iter': 100,
        'name': 'large_matrix'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Low tolerance
    input_dict_10 = {
        'operator': create_spd_matrix((3, 3), np.float64),
        'rhs': np.array([0.1, -0.5, 1.2], dtype=np.float64),
        'preconditioner': None,
        'x': np.zeros(3, dtype=np.float64),
        'tol': 1e-10,
        'max_iter': 50,
        'name': 'low_tolerance'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.linalg.experimental.conjugate_gradient"] = get_conjugate_gradient_inputs()

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
