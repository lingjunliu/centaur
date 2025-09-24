
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_MatrixDiagPart_inputs():
    list_of_inputs = []

    # Input 1: Basic 2x2 matrix
    input_matrix = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict = {"input": input_matrix, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3x3 matrix
    input_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    input_dict = {"input": input_matrix, "name": "matrix_diag"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Rectangular matrix (2x3)
    input_matrix = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    input_dict = {"input": input_matrix, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Batched matrices (2x2x2)
    input_matrix = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    input_dict = {"input": input_matrix, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative values in the matrix
    input_matrix = np.array([[-1, 2], [3, -4]], dtype=np.int32)
    input_dict = {"input": input_matrix, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger dimensions
    input_matrix = np.random.randint(0, 10, size=(5, 5), dtype=np.int32)
    input_dict = {"input": input_matrix, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4-D Tensor
    input_matrix = np.random.randint(0, 10, size=(2, 3, 3, 3), dtype=np.int32)
    input_dict = {"input": input_matrix, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Batched matrices with different dimensions (2x3x3)
    input_matrix = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]]], dtype=np.int64)
    input_dict = {"input": input_matrix, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Rectangular matrix (3x2)
    input_matrix = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    input_dict = {"input": input_matrix, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Batched matrices (3x2x2)
    input_matrix = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)
    input_dict = {"input": input_matrix, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.MatrixDiagPart"] = tf_raw_ops_MatrixDiagPart_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.MatrixDiagPart' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MatrixDiagPart'.")

check_valid('tf.raw_ops.MatrixDiagPart', generated_inputs['tf.raw_ops.MatrixDiagPart'], lib="tf", suffix=0)
