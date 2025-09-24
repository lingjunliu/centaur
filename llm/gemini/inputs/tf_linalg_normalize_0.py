
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_tf_linalg_normalize_inputs():
    """
    Generates a list of valid inputs for the tf.linalg.normalize function.
    """
    list_of_inputs = []

    # All inputs will use a 2-element tuple for 'axis' and a string for 'ord'
    # to strictly adhere to the provided signature and avoid the previous errors.
    # This means we only generate matrix norms with 'fro' or 'euclidean' order.

    # === Case 1: Basic 2D matrix, float32, Frobenius norm ===
    tensor1 = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    input_dict1 = {
        'tensor': tensor1,
        'ord': 'fro',
        'axis': (0, 1),
        'name': 'fro_2d_float32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # === Case 2: Basic 2D matrix, float64, Euclidean norm ===
    tensor2 = np.array([[5., 6.], [7., 8.]], dtype=np.float64)
    input_dict2 = {
        'tensor': tensor2,
        'ord': 'euclidean',
        'axis': (0, 1),
        'name': 'euclidean_2d_float64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # === Case 3: Batch of matrices (3D), Frobenius norm ===
    tensor3 = np.arange(1, 13, dtype=np.float32).reshape(2, 2, 3)
    input_dict3 = {
        'tensor': tensor3,
        'ord': 'fro',
        'axis': (1, 2),
        'name': 'fro_3d_batch'
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # === Case 4: Batch of matrices (3D), Euclidean norm, negative values & axis ===
    tensor4 = np.array([[[1., -2.], [3., 4.]], [[-5., 6.], [7., -8.]]], dtype=np.float32)
    input_dict4 = {
        'tensor': tensor4,
        'ord': 'euclidean',
        'axis': (-2, -1),
        'name': 'euclidean_3d_neg_vals_axis'
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # === Case 5: 2D Complex tensor, complex64, Frobenius norm ===
    tensor5 = np.array([[1+1j, 2-2j], [3+3j, 4-4j]], dtype=np.complex64)
    input_dict5 = {
        'tensor': tensor5,
        'ord': 'fro',
        'axis': (0, 1),
        'name': 'fro_2d_complex64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # === Case 6: 3D Complex tensor, complex128, Euclidean norm ===
    tensor6 = np.array([[[1+1j, 2j], [3, 4-4j]]], dtype=np.complex128)
    input_dict6 = {
        'tensor': tensor6,
        'ord': 'euclidean',
        'axis': (1, 2),
        'name': 'euclidean_3d_complex128'
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # === Case 7: Higher-dimensional tensor (4D), Frobenius norm ===
    tensor7 = np.arange(1, 25, dtype=np.float32).reshape(2, 2, 2, 3)
    input_dict7 = {
        'tensor': tensor7,
        'ord': 'fro',
        'axis': (2, 3),
        'name': 'fro_4d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # === Case 8: Higher-dimensional tensor (4D), Euclidean, non-adjacent axes ===
    tensor8 = np.arange(1, 25, dtype=np.float64).reshape(2, 3, 2, 2)
    input_dict8 = {
        'tensor': tensor8,
        'ord': 'euclidean',
        'axis': (1, 3),
        'name': 'euclidean_4d_non_adj_axis'
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # === Case 9: Matrix with a zero-norm sub-matrix (edge case) ===
    tensor9 = np.array([[[1., 2.], [3., 4.]], [[0., 0.], [0., 0.]]], dtype=np.float32)
    input_dict9 = {
        'tensor': tensor9,
        'ord': 'fro',
        'axis': (1, 2),
        'name': 'zero_submatrix'
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # === Case 10: Non-square matrix (2x5), float64 ===
    tensor10 = np.random.rand(2, 5).astype(np.float64)
    input_dict10 = {
        'tensor': tensor10,
        'ord': 'fro',
        'axis': (0, 1),
        'name': 'fro_2x5_float64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    # === Case 11: Non-square matrix (5x2), float32 batch ===
    tensor11 = np.random.rand(3, 5, 2).astype(np.float32)
    input_dict11 = {
        'tensor': tensor11,
        'ord': 'euclidean',
        'axis': (1, 2),
        'name': 'euclidean_5x2_batch_float32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict11))
    
    # === Case 12: Large matrix ===
    tensor12 = np.arange(100, dtype=np.float32).reshape(10, 10)
    input_dict12 = {
        'tensor': tensor12,
        'ord': 'fro',
        'axis': (0, 1),
        'name': 'large_matrix_fro'
    }
    list_of_inputs.append(copy.deepcopy(input_dict12))

    return list_of_inputs

generated_inputs["tf.linalg.normalize"] = get_tf_linalg_normalize_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.linalg.normalize' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.normalize'.")

check_valid('tf.linalg.normalize', generated_inputs['tf.linalg.normalize'], lib="tf", suffix=0)
