
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_lu_inputs():
    list_of_inputs = []

    # Input 1: Simple float32 matrix
    input_matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"input": input_matrix, "output_idx_type": tf.int32, "name": None}
    list_of_inputs.append(input_dict)

    # Input 2: float64 matrix with negative values
    input_matrix = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float64)
    input_dict = {"input": input_matrix, "output_idx_type": tf.int64, "name": "lu_decomposition"}
    list_of_inputs.append(input_dict)

    # Input 3: Complex64 matrix
    input_matrix = np.array([[1.0 + 1j, 2.0], [3.0, 4.0 - 1j]], dtype=np.complex64)
    input_dict = {"input": input_matrix, "output_idx_type": tf.int64, "name": None}
    list_of_inputs.append(input_dict)

    # Input 4: Complex128 matrix
    input_matrix = np.array([[1.0 + 1j, 2.0], [3.0, 4.0 - 1j]], dtype=np.complex128)
    input_dict = {"input": input_matrix, "output_idx_type": tf.int32, "name": "complex_lu"}
    list_of_inputs.append(input_dict)

    # Input 5: 3D float32 tensor
    input_matrix = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict = {"input": input_matrix, "output_idx_type": tf.int64, "name": None}
    list_of_inputs.append(input_dict)

    # Input 6: Different matrix size (3x3) with float32
    input_matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]], dtype=np.float32)
    input_dict = {"input": input_matrix, "output_idx_type": tf.int32, "name": "lu_3x3"}
    list_of_inputs.append(input_dict)

    # Input 7: Matrix with zero values (but still invertible) with float64
    input_matrix = np.array([[1.0, 0.0], [0.0, 4.0]], dtype=np.float64)
    input_dict = {"input": input_matrix, "output_idx_type": tf.int64, "name": None}
    list_of_inputs.append(input_dict)

    # Input 8: larger float32 matrix with negative values and different output_idx_type
    input_matrix = np.array([[1.0, -2.0], [5.0, 6.0]], dtype=np.float32)
    input_dict = {"input": input_matrix, "output_idx_type": tf.int32, "name": None}
    list_of_inputs.append(input_dict)

    # Input 9: another 3D example
    input_matrix = np.array([[[1.0, 2.0], [3.0, 4.0]], [[-1.0, -2.0], [-3.0, -4.0]]], dtype=np.float64)
    input_dict = {"input": input_matrix, "output_idx_type": tf.int64, "name": None}
    list_of_inputs.append(input_dict)

    # Input 10: Example with more significant digits.
    input_matrix = np.array([[1.001, 2.002], [3.003, 4.004]], dtype=np.float32)
    input_dict = {"input": input_matrix, "output_idx_type": tf.int32, "name": None}
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.lu"] = tf_linalg_lu_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.lu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.lu'.")

check_valid('tf.linalg.lu', generated_inputs['tf.linalg.lu'], lib="tf", suffix=0)
