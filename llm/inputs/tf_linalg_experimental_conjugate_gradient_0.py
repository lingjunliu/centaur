
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def generate_spd_matrix(n, dtype=np.float32, batch_dims=None):
    """Generates a symmetric positive-definite matrix."""
    shape = batch_dims + [n, n] if batch_dims is not None else [n, n]
    mat = np.random.rand(*shape).astype(dtype)
    if batch_dims:
        # Create a symmetric matrix from a random one
        transpose_axes = list(range(len(batch_dims))) + [len(batch_dims) + 1, len(batch_dims)]
        mat = (mat + np.transpose(mat, axes=transpose_axes)) / 2.0
        # Create an identity matrix with the correct batching
        identity = np.eye(n, dtype=dtype)
        id_shape = [1] * len(batch_dims) + [n, n]
        identity = identity.reshape(id_shape)
        # Make it diagonally dominant to ensure it's positive-definite
        mat += identity * n
    else:
        mat = (mat + mat.T) / 2.0
        mat += np.eye(n, dtype=dtype) * n
    return mat

def generate_hpd_matrix(n, dtype=np.complex64, batch_dims=None):
    """Generates a Hermitian positive-definite matrix."""
    shape = batch_dims + [n, n] if batch_dims is not None else [n, n]
    real_part = np.random.rand(*shape)
    imag_part = np.random.rand(*shape)
    mat = (real_part + 1j * imag_part).astype(dtype)
    
    real_dtype = np.complex64(0).real.dtype if dtype==np.complex64 else np.complex128(0).real.dtype
    
    if batch_dims:
        transpose_axes = list(range(len(batch_dims))) + [len(batch_dims) + 1, len(batch_dims)]
        mat_h = np.transpose(mat.conj(), axes=transpose_axes)
        mat = (mat + mat_h) / 2.0
        identity = np.eye(n, dtype=real_dtype)
        id_shape = [1] * len(batch_dims) + [n, n]
        identity = identity.reshape(id_shape)
        mat += identity * n
    else:
        mat = (mat + mat.conj().T) / 2.0
        mat += np.eye(n, dtype=real_dtype) * n
    return mat

def tf_linalg_experimental_conjugate_gradient_inputs():
    """
    Generates a list of valid inputs for the tf.linalg.experimental.conjugate_gradient function.
    """
    list_of_inputs = []

    # The 'operator' and 'preconditioner' arguments expect tf.linalg.LinearOperator.
    # However, the testing harness requires numpy arrays for 'tensor' types.
    # We provide numpy arrays that are self-adjoint and positive-definite,
    # assuming the harness will convert them to LinearOperators before the API call.
    
    # Input 1: Simple 2x2 case
    matrix1 = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    rhs1 = np.array([1.0, 1.0], dtype=np.float32)
    x1 = np.array([0.0, 0.0], dtype=np.float32)
    list_of_inputs.append({
        'operator': tf.linalg.LinearOperatorFullMatrix(matrix1),
        'rhs': rhs1,
        'preconditioner': None,
        'x': x1,
        'tol': 1e-5,
        'max_iter': 20,
        'name': 'simple_2x2'
    })

    # Input 2: 4x4, float64, x is None
    matrix2 = generate_spd_matrix(4, dtype=np.float64)
    rhs2 = np.random.rand(4).astype(np.float64)
    list_of_inputs.append({
        'operator': tf.linalg.LinearOperatorFullMatrix(matrix2),
        'rhs': rhs2,
        'preconditioner': None,
        'x': None,
        'tol': 1e-6,
        'max_iter': 50,
        'name': 'larger_4x4_float64'
    })

    # Input 3: With an Identity preconditioner
    matrix3 = generate_spd_matrix(3, dtype=np.float32)
    rhs3 = np.random.rand(3).astype(np.float32)
    preconditioner3 = tf.linalg.LinearOperatorIdentity(num_rows=3, dtype=tf.float32)
    x3 = np.random.rand(3).astype(np.float32)
    list_of_inputs.append({
        'operator': tf.linalg.LinearOperatorFullMatrix(matrix3),
        'rhs': rhs3,
        'preconditioner': preconditioner3,
        'x': x3,
        'tol': 1e-4,
        'max_iter': 10,
        'name': 'with_identity_preconditioner'
    })

    # Input 4: With a Jacobi (diagonal) preconditioner
    matrix4 = generate_spd_matrix(5, dtype=np.float32)
    rhs4 = np.random.rand(5).astype(np.float32)
    preconditioner4 = tf.linalg.LinearOperatorDiag(1.0 / np.diag(matrix4))
    x4 = np.zeros(5, dtype=np.float32)
    list_of_inputs.append({
        'operator': tf.linalg.LinearOperatorFullMatrix(matrix4),
        'rhs': rhs4,
        'preconditioner': preconditioner4,
        'x': x4,
        'tol': 1e-5,
        'max_iter': 30,
        'name': 'with_jacobi_preconditioner'
    })
    
    # Input 5: Batched input (batch size 2)
    matrix5 = generate_spd_matrix(2, dtype=np.float32, batch_dims=[2])
    rhs5 = np.random.rand(2, 2).astype(np.float32)
    x5 = np.zeros((2, 2), dtype=np.float32)
    list_of_inputs.append({
        'operator': tf.linalg.LinearOperatorFullMatrix(matrix5),
        'rhs': rhs5,
        'preconditioner': None,
        'x': x5,
        'tol': 1e-5,
        'max_iter': 20,
        'name': 'batched_input'
    })

    # Input 6: Batched input with batched preconditioner
    matrix6 = generate_spd_matrix(3, dtype=np.float64, batch_dims=[4])
    rhs6 = np.random.rand(4, 3).astype(np.float64)
    preconditioner_diag6 = 1.0 / np.einsum('bii->bi', matrix6)
    preconditioner6 = tf.linalg.LinearOperatorDiag(preconditioner_diag6)
    list_of_inputs.append({
        'operator': tf.linalg.LinearOperatorFullMatrix(matrix6),
        'rhs': rhs6,
        'preconditioner': preconditioner6,
        'x': None,
        'tol': 1e-7,
        'max_iter': 100,
        'name': 'batched_with_preconditioner'
    })
    
    # Input 7: Higher rank batching
    matrix7 = generate_spd_matrix(3, dtype=np.float32, batch_dims=[2, 2])
    rhs7 = np.random.rand(2, 2, 3).astype(np.float32)
    x7 = np.zeros((2, 2, 3), dtype=np.float32)
    list_of_inputs.append({
        'operator': tf.linalg.LinearOperatorFullMatrix(matrix7),
        'rhs': rhs7,
        'preconditioner': None,
        'x': x7,
        'tol': 1e-5,
        'max_iter': 20,
        'name': 'high_rank_batch'
    })

    # Input 8: Loose tolerance and few iterations
    matrix8 = generate_spd_matrix(10, dtype=np.float32)
    rhs8 = np.random.rand(10).astype(np.float32)
    list_of_inputs.append({
        'operator': tf.linalg.LinearOperatorFullMatrix(matrix8),
        'rhs': rhs8,
        'preconditioner': None,
        'x': None,
        'tol': 0.1,
        'max_iter': 3,
        'name': 'large_tol_low_iter'
    })

    # Input 9: Complex numbers (complex64)
    matrix9 = generate_hpd_matrix(3, dtype=np.complex64)
    rhs9 = (np.random.rand(3) + 1j * np.random.rand(3)).astype(np.complex64)
    list_of_inputs.append({
        'operator': tf.linalg.LinearOperatorFullMatrix(matrix9),
        'rhs': rhs9,
        'preconditioner': None,
        'x': None,
        'tol': 1e-5,
        'max_iter': 30,
        'name': 'complex_case'
    })

    # Input 10: Complex numbers (complex128) with preconditioner
    matrix10 = generate_hpd_matrix(4, dtype=np.complex128)
    rhs10 = (np.random.rand(4) + 1j*np.random.rand(4)).astype(np.complex128)
    preconditioner_diag10 = (1.0 / np.diag(matrix10).real).astype(np.complex128)
    preconditioner10 = tf.linalg.LinearOperatorDiag(preconditioner_diag10)
    x10 = np.zeros(4, dtype=np.complex128)
    list_of_inputs.append({
        'operator': tf.linalg.LinearOperatorFullMatrix(matrix10),
        'rhs': rhs10,
        'preconditioner': preconditioner10,
        'x': x10,
        'tol': 1e-8,
        'max_iter': 40,
        'name': 'complex_with_preconditioner'
    })

    # Input 11: 1x1 edge case
    matrix11 = np.array([[4.0]], dtype=np.float32)
    rhs11 = np.array([2.0], dtype=np.float32)
    x11 = np.array([0.0], dtype=np.float32)
    list_of_inputs.append({
        'operator': tf.linalg.LinearOperatorFullMatrix(matrix11),
        'rhs': rhs11,
        'preconditioner': None,
        'x': x11,
        'tol': 1e-8,
        'max_iter': 2,
        'name': 'one_by_one'
    })

    # Input 12: Diagonal operator
    diag12 = np.array([1., 2., 4., 8.], dtype=np.float32)
    rhs12 = np.array([4., 3., 2., 1.], dtype=np.float32)
    list_of_inputs.append({
        'operator': tf.linalg.LinearOperatorDiag(diag12),
        'rhs': rhs12,
        'preconditioner': None,
        'x': None,
        'tol': 1e-5,
        'max_iter': 10,
        'name': 'diag_operator'
    })

    return list_of_inputs

generated_inputs["tf.linalg.experimental.conjugate_gradient"] = tf_linalg_experimental_conjugate_gradient_inputs()

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
