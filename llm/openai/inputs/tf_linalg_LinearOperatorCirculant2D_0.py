
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_linalg_LinearOperatorCirculant2D_inputs():
    rng = np.random.default_rng(123)
    list_of_inputs = []

    def hermitize(arr):
        a = np.array(arr, dtype=np.complex64, copy=True)
        N0, N1 = a.shape[-2], a.shape[-1]
        out = np.empty_like(a)
        it = np.ndindex(a.shape[:-2])
        for idx in it:
            A = a[idx]
            H = np.empty_like(A)
            for n0 in range(N0):
                for n1 in range(N1):
                    m0 = (-n0) % N0
                    m1 = (-n1) % N1
                    H[n0, n1] = 0.5 * (A[n0, n1] + np.conj(A[m0, m1]))
            out[idx] = H
        return out

    spectrum = np.array([[1.5, 2.0, 3.25],
                         [4.5, 5.0, 6.75]], dtype=np.float32)
    input_dict = {
        "spectrum": spectrum,
        "input_output_dtype": np.complex64,
        "is_non_singular": True,
        "is_self_adjoint": True,
        "is_positive_definite": True,
        "is_square": True,
        "name": "circ2d_real_pos_2x3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    spectrum = np.array([[1+2j, -0.5+0.1j, 0.0-1j],
                         [2-3j,  0.3+0.7j, 2.2+0.0j],
                         [-1+0.5j, 0.7-0.4j, -2.1+1.1j]], dtype=np.complex64)
    input_dict = {
        "spectrum": spectrum,
        "input_output_dtype": np.complex64,
        "is_non_singular": False,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": True,
        "name": "circ2d_cplx_3x3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    base = rng.normal(size=(2, 3, 3)) + 1j * rng.normal(size=(2, 3, 3))
    spectrum = hermitize(base.astype(np.complex64))
    input_dict = {
        "spectrum": spectrum,
        "input_output_dtype": np.float64,
        "is_non_singular": False,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": True,
        "name": "circ2d_batched_hermitian_2x3x3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    spectrum = np.array([[0., 1., 0., 2.],
                         [3., 0., 4., 0.],
                         [0., 5., 0., 6.],
                         [7., 0., 8., 0.]], dtype=np.float32)
    input_dict = {
        "spectrum": spectrum,
        "input_output_dtype": np.complex64,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "name": "circ2d_real_with_zeros_4x4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    spectrum = np.ones((5, 2, 2), dtype=np.float64)
    input_dict = {
        "spectrum": spectrum,
        "input_output_dtype": np.complex128,
        "is_non_singular": True,
        "is_self_adjoint": True,
        "is_positive_definite": True,
        "is_square": True,
        "name": "circ2d_batched_ones_5x2x2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    real = rng.normal(size=(2, 4, 5)).astype(np.float32)
    imag = rng.normal(size=(2, 4, 5)).astype(np.float32)
    spectrum = real + 1j * imag
    input_dict = {
        "spectrum": spectrum.astype(np.complex64),
        "input_output_dtype": np.dtype(np.complex64),
        "is_non_singular": False,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": True,
        "name": "circ2d_cplx_batched_2x4x5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    kernel = np.array([[1., 2., 1.],
                       [5., -1., 1.]], dtype=np.float32)
    spectrum = np.fft.fft2(kernel).astype(np.complex64)
    input_dict = {
        "spectrum": spectrum,
        "input_output_dtype": np.float32,
        "is_non_singular": False,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": True,
        "name": "circ2d_fft_kernel_2x3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    spectrum = np.array([[1., 2., 3.],
                         [4., 5., 6.],
                         [7., 8., 9.]], dtype=np.float64)
    input_dict = {
        "spectrum": spectrum,
        "input_output_dtype": np.complex64,
        "is_non_singular": True,
        "is_self_adjoint": True,
        "is_positive_definite": True,
        "is_square": True,
        "name": "circ2d_real_pd_3x3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    spectrum = np.array([[2.0]], dtype=np.float32)
    input_dict = {
        "spectrum": spectrum,
        "input_output_dtype": np.dtype('complex64'),
        "is_non_singular": True,
        "is_self_adjoint": True,
        "is_positive_definite": True,
        "is_square": True,
        "name": "circ2d_scalar_1x1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    spectrum = np.abs(rng.integers(low=1, high=5, size=(2, 3, 2, 2))).astype(np.float32)
    input_dict = {
        "spectrum": spectrum,
        "input_output_dtype": np.complex128,
        "is_non_singular": True,
        "is_self_adjoint": True,
        "is_positive_definite": True,
        "is_square": True,
        "name": "circ2d_multibatch_pos_2x3x2x2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    spectrum = np.array([[-1.0, -2.0],
                         [-3.0, -4.0]], dtype=np.float32)
    input_dict = {
        "spectrum": spectrum,
        "input_output_dtype": np.dtype(np.complex64),
        "is_non_singular": True,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "name": "circ2d_real_negative_2x2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    spectrum = rng.normal(size=(4, 3)).astype(np.float64)
    input_dict = {
        "spectrum": spectrum,
        "input_output_dtype": np.complex64,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "name": "circ2d_real_4x3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.linalg.LinearOperatorCirculant2D"] = tf_linalg_LinearOperatorCirculant2D_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.linalg.LinearOperatorCirculant2D' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorCirculant2D'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.linalg.LinearOperatorCirculant2D', generated_inputs['tf.linalg.LinearOperatorCirculant2D'], lib="tf", suffix=0)
