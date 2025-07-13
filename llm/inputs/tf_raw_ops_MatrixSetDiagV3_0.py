
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_matrix_set_diag_v3_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]).astype(np.int32)
    diagonal_tensor = np.array([[10, 11, 12]]).astype(np.int32)
    k_tensor = np.array(0).astype(np.int32)
    align_str = "RIGHT_LEFT"

    input_dict = {
        "input": input_tensor,
        "diagonal": diagonal_tensor,
        "k": k_tensor,
        "align": align_str,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]).astype(np.float32)
    diagonal_tensor = np.array([[10.0, 11.0, 12.0]]).astype(np.float32)
    k_tensor = np.array(1).astype(np.int32)
    align_str = "LEFT_RIGHT"
    input_dict = {
        "input": input_tensor,
        "diagonal": diagonal_tensor,
        "k": k_tensor,
        "align": align_str,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]).astype(np.int64)
    diagonal_tensor = np.array([[10, 11, 12]]).astype(np.int64)
    k_tensor = np.array(-1).astype(np.int32)
    align_str = "LEFT_LEFT"
    input_dict = {
        "input": input_tensor,
        "diagonal": diagonal_tensor,
        "k": k_tensor,
        "align": align_str,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]).astype(np.float64)
    diagonal_tensor = np.array([[10.0, 11.0, 12.0]]).astype(np.float64)
    k_tensor = np.array(0).astype(np.int32)
    align_str = "RIGHT_RIGHT"
    input_dict = {
        "input": input_tensor,
        "diagonal": diagonal_tensor,
        "k": k_tensor,
        "align": align_str,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).astype(np.int32)
    diagonal_tensor = np.array([[10, 11], [12, 13]]).astype(np.int32)
    k_tensor = np.array(0).astype(np.int32)
    align_str = "RIGHT_LEFT"
    input_dict = {
        "input": input_tensor,
        "diagonal": diagonal_tensor,
        "k": k_tensor,
        "align": align_str,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).astype(np.float32)
    diagonal_tensor = np.array([[10.0, 11.0], [12.0, 13.0]]).astype(np.float32)
    k_tensor = np.array([0, 1]).astype(np.int32)
    align_str = "LEFT_RIGHT"
    input_dict = {
        "input": input_tensor,
        "diagonal": diagonal_tensor,
        "k": k_tensor,
        "align": align_str,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).astype(np.int64)
    diagonal_tensor = np.array([[10, 11], [12, 13]]).astype(np.int64)
    k_tensor = np.array([-1, 0]).astype(np.int32)
    align_str = "LEFT_LEFT"
    input_dict = {
        "input": input_tensor,
        "diagonal": diagonal_tensor,
        "k": k_tensor,
        "align": align_str,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).astype(np.float64)
    diagonal_tensor = np.array([[10.0, 11.0], [12.0, 13.0]]).astype(np.float64)
    k_tensor = np.array([-1, 1]).astype(np.int32)
    align_str = "RIGHT_RIGHT"
    input_dict = {
        "input": input_tensor,
        "diagonal": diagonal_tensor,
        "k": k_tensor,
        "align": align_str,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9
    input_tensor = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]]).astype(np.int32)
    diagonal_tensor = np.array([[[13, 14], [15, 16], [17, 18]]]).astype(np.int32)
    k_tensor = np.array([-1, 1]).astype(np.int32)
    align_str = "RIGHT_LEFT"
    input_dict = {
        "input": input_tensor,
        "diagonal": diagonal_tensor,
        "k": k_tensor,
        "align": align_str,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]]).astype(np.float32)
    diagonal_tensor = np.array([[[13.0, 14.0], [15.0, 16.0], [17.0, 18.0]]]).astype(np.float32)
    k_tensor = np.array([-2, 0]).astype(np.int32)
    align_str = "LEFT_RIGHT"
    input_dict = {
        "input": input_tensor,
        "diagonal": diagonal_tensor,
        "k": k_tensor,
        "align": align_str,
        "name": None
    }
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
