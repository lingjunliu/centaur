
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_matrix_set_diag_v3_inputs():
    list_of_inputs = []

    # Input 1: Basic example with main diagonal
    input_matrix = np.array([[[7, 7, 7], [7, 7, 7], [7, 7, 7]]], dtype=np.int32)
    diagonal = np.array([[1, 2, 3]], dtype=np.int32)
    k = np.array(0, dtype=np.int32)
    input_dict = {"input": input_matrix, "diagonal": diagonal, "k": k, "align": "RIGHT_LEFT", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Superdiagonal
    input_matrix = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=np.int32)
    diagonal = np.array([[10, 11]], dtype=np.int32)
    k = np.array(1, dtype=np.int32)
    input_dict = {"input": input_matrix, "diagonal": diagonal, "k": k, "align": "RIGHT_LEFT", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Subdiagonal
    input_matrix = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=np.int32)
    diagonal = np.array([[10, 11]], dtype=np.int32)
    k = np.array(-1, dtype=np.int32)
    input_dict = {"input": input_matrix, "diagonal": diagonal, "k": k, "align": "RIGHT_LEFT", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Band of diagonals
    input_matrix = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]], dtype=np.int32)
    diagonal = np.array([[[13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24]]], dtype=np.int32)
    k = np.array([-1, 2], dtype=np.int32)
    input_dict = {"input": input_matrix, "diagonal": diagonal, "k": k, "align": "RIGHT_LEFT", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different alignment
    input_matrix = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]], dtype=np.int32)
    diagonal = np.array([[[13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24]]], dtype=np.int32)
    k = np.array([-1, 2], dtype=np.int32)
    input_dict = {"input": input_matrix, "diagonal": diagonal, "k": k, "align": "LEFT_RIGHT", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: k is scalar
    input_matrix = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=np.int32)
    diagonal = np.array([[10, 11, 12]], dtype=np.int32)
    k = np.array(0, dtype=np.int32)
    input_dict = {"input": input_matrix, "diagonal": diagonal, "k": k, "align": "RIGHT_LEFT", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Multiple batches
    input_matrix = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.int32)
    diagonal = np.array([[13, 14], [15, 16]], dtype=np.int32)
    k = np.array(0, dtype=np.int32)
    input_dict = {"input": input_matrix, "diagonal": diagonal, "k": k, "align": "RIGHT_LEFT", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different data type
    input_matrix = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    diagonal = np.array([[5.0, 6.0]], dtype=np.float32)
    k = np.array(0, dtype=np.int32)
    input_dict = {"input": input_matrix, "diagonal": diagonal, "k": k, "align": "RIGHT_LEFT", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: k[0] == k[1]
    input_matrix = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=np.int32)
    diagonal = np.array([[10, 11, 12]], dtype=np.int32)
    k = np.array([0, 0], dtype=np.int32)
    input_dict = {"input": input_matrix, "diagonal": diagonal, "k": k, "align": "RIGHT_LEFT", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: More complex band
    input_matrix = np.array([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20]]], dtype=np.int32)
    diagonal = np.array([[[21, 22, 23, 24], [25, 26, 27, 28], [29, 30, 31, 32], [33, 34, 35, 36], [37, 38, 39, 40]]], dtype=np.int32)
    k = np.array([-2, 2], dtype=np.int32)

    input_dict = {"input": input_matrix, "diagonal": diagonal, "k": k, "align": "RIGHT_LEFT", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.MatrixSetDiagV3"] = tf_raw_ops_matrix_set_diag_v3_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.MatrixSetDiagV3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MatrixSetDiagV3'.")

check_valid('tf.raw_ops.MatrixSetDiagV3', generated_inputs['tf.raw_ops.MatrixSetDiagV3'], lib="tf", suffix=0)
