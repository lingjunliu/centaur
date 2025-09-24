
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_slogdet_inputs():
    list_of_inputs = []

    # Input 1: Basic 2x2 matrix
    matrix1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"input": matrix1, "name": "basic_matrix"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Batch of 2x2 matrices
    matrix2 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict = {"input": matrix2, "name": "batch_matrix"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3x3 matrix with negative values
    matrix3 = np.array([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0], [7.0, -8.0, 9.0]], dtype=np.float32)
    input_dict = {"input": matrix3, "name": "negative_matrix"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Batch of 3x3 matrices with complex numbers
    matrix4 = np.array([[[1+1j, 2+2j, 3+3j], [4+4j, 5+5j, 6+6j], [7+7j, 8+8j, 9+9j]], [[9+9j, 8+8j, 7+7j], [6+6j, 5+5j, 4+4j], [3+3j, 2+2j, 1+1j]]], dtype=np.complex64)
    input_dict = {"input": matrix4, "name": "complex_matrix"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4x4 matrix
    matrix6 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]], dtype=np.float64)
    input_dict = {"input": matrix6, "name": "4x4_matrix"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1x1 matrix
    matrix7 = np.array([[5.0]], dtype=np.float32)
    input_dict = {"input": matrix7, "name": "1x1_matrix"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Batch of 1x1 matrices
    matrix8 = np.array([[[5.0]], [[10.0]]], dtype=np.float32)
    input_dict = {"input": matrix8, "name": "batch_1x1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2x2 matrix with double precision complex numbers
    matrix9 = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex128)
    input_dict = {"input": matrix9, "name": "complex128_matrix"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Batch of 2x2 matrices with some zero values
    matrix10 = np.array([[[1.0, 0.0], [0.0, 4.0]], [[5.0, 0.0], [0.0, 8.0]]], dtype=np.float32)
    input_dict = {"input": matrix10, "name": "batch_zero"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: A valid 2x2 matrix
    matrix11 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    input_dict = {"input": matrix11, "name": "valid_matrix"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.slogdet"] = tf_linalg_slogdet_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.slogdet' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.slogdet'.")

check_valid('tf.linalg.slogdet', generated_inputs['tf.linalg.slogdet'], lib="tf", suffix=0)
