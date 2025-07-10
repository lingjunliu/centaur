
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_tensor_diag_part_inputs():
    list_of_inputs = []

    # Input 1: 2x2 matrix
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 4x4 matrix
    input_tensor = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.float32)
    input_dict = {"input": input_tensor, "name": "matrix_4x4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2x2x2x2 tensor
    input_tensor = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]], dtype=np.int64)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative values
    input_tensor = np.array([[-1, 2], [3, -4]], dtype=np.int32)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger matrix
    input_tensor = np.random.rand(5, 5).astype(np.float32)
    input_dict = {"input": input_tensor, "name": "large_matrix"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Complex numbers
    input_tensor = np.array([[1+1j, 2-2j], [3+3j, 4-4j]], dtype=np.complex64)
    input_dict = {"input": input_tensor, "name": "complex_matrix"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Another 2k tensor (2x2x2x2)
    input_tensor = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]], dtype=np.int32)
    input_dict = {"input": input_tensor, "name": "tensor_2x3x2x3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Float64
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 3x3 matrix
    input_tensor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 4x4x4x4 matrix

    input_tensor = np.random.rand(2, 2, 2, 2).astype(np.float32)
    input_dict = {"input": input_tensor, "name": "matrix_4x4x4x4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.tensor_diag_part"] = tf_linalg_tensor_diag_part_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.tensor_diag_part' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.tensor_diag_part'.")

check_valid('tf.linalg.tensor_diag_part', generated_inputs['tf.linalg.tensor_diag_part'], lib="tf", suffix=0)
