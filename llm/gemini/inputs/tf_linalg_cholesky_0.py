
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_cholesky_inputs():
    list_of_inputs = []

    # Input 1: Simple 2x2 matrix
    input_matrix = np.array([[4.0, 1.0], [1.0, 4.0]], dtype=np.float32)
    input_dict = {"input": input_matrix, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Batch of 2x2 matrices
    input_matrix1 = np.array([[4.0, 1.0], [1.0, 4.0]], dtype=np.float64)
    input_matrix2 = np.array([[9.0, 3.0], [3.0, 9.0]], dtype=np.float64)
    input_tensor = np.array([input_matrix1, input_matrix2])
    input_dict = {"input": input_tensor, "name": "batch_cholesky"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3x3 matrix
    input_matrix = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]], dtype=np.float32)
    input_dict = {"input": input_matrix, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Larger matrix (4x4)
    input_matrix = np.array([[4.0, 1.0, 0.0, 0.0], [1.0, 4.0, 1.0, 0.0], [0.0, 1.0, 4.0, 1.0], [0.0, 0.0, 1.0, 4.0]], dtype=np.float64)
    input_dict = {"input": input_matrix, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex 2x2 matrix
    input_matrix = np.array([[4 + 0j, 1 + 0j], [1 + 0j, 4 + 0j]], dtype=np.complex64)
    input_dict = {"input": input_matrix, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Complex batch of 2x2
    input_matrix1 = np.array([[4 + 0j, 1 + 0j], [1 + 0j, 4 + 0j]], dtype=np.complex128)
    input_matrix2 = np.array([[9 + 0j, 3 + 0j], [3 + 0j, 9 + 0j]], dtype=np.complex128)
    input_tensor = np.array([input_matrix1, input_matrix2])
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D tensor (batch size 3, 2x2 matrices) float32
    input_matrix1 = np.array([[4.0, 1.0], [1.0, 4.0]], dtype=np.float32)
    input_matrix2 = np.array([[9.0, 3.0], [3.0, 9.0]], dtype=np.float32)
    input_matrix3 = np.array([[16.0, 4.0], [4.0, 16.0]], dtype=np.float32)
    input_tensor = np.array([input_matrix1, input_matrix2, input_matrix3])
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger batch size with larger matrices (3x3)
    input_matrix1 = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]], dtype=np.float64)
    input_matrix2 = np.array([[5.0, -2.0, 0.0], [-2.0, 5.0, -2.0], [0.0, -2.0, 5.0]], dtype=np.float64)
    input_tensor = np.array([input_matrix1, input_matrix2])
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Name provided
    input_matrix = np.array([[4.0, 1.0], [1.0, 4.0]], dtype=np.float32)
    input_dict = {"input": input_matrix, "name": "my_cholesky"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.cholesky"] = tf_linalg_cholesky_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.cholesky' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.cholesky'.")

check_valid('tf.linalg.cholesky', generated_inputs['tf.linalg.cholesky'], lib="tf", suffix=0)
