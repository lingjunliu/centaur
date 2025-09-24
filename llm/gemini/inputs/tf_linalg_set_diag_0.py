
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_set_diag_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.int32)
    diagonal_tensor = np.array([10, 11, 12]).astype(np.int32)
    input_dict = {"input": input_tensor, "diagonal": diagonal_tensor, "name": "diag1", "k": np.int32(0), "align": "RIGHT_LEFT"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.int32)
    diagonal_tensor = np.array([10, 11]).astype(np.int32)
    input_dict = {"input": input_tensor, "diagonal": diagonal_tensor, "name": "diag2", "k": np.int32(1), "align": "RIGHT_LEFT"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.int32)
    diagonal_tensor = np.array([10, 11]).astype(np.int32)
    input_dict = {"input": input_tensor, "diagonal": diagonal_tensor, "name": "diag3", "k": np.int32(-1), "align": "RIGHT_LEFT"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).astype(np.int32)
    diagonal_tensor = np.array([[10, 11], [12, 13]]).astype(np.int32)
    input_dict = {"input": input_tensor, "diagonal": diagonal_tensor, "name": "diag4", "k": np.int32(0), "align": "RIGHT_LEFT"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).astype(np.int32)
    diagonal_tensor = np.array([[10], [12]]).astype(np.int32)
    input_dict = {"input": input_tensor, "diagonal": diagonal_tensor, "name": "diag5", "k": np.int32(1), "align": "RIGHT_LEFT"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).astype(np.int32)
    diagonal_tensor = np.array([[10], [12]]).astype(np.int32)
    input_dict = {"input": input_tensor, "diagonal": diagonal_tensor, "name": "diag6", "k": np.int32(-1), "align": "RIGHT_LEFT"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]).astype(np.int32)
    diagonals = np.array([[[9, 1, 0], [6, 5, 8], [1, 2, 3], [0, 4, 5]]]).astype(np.int32)
    input_dict = {"input": input_tensor, "diagonal": diagonals[0], "name": "diag7", "k": (np.int32(-1), np.int32(2)), "align": "RIGHT_LEFT"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]).astype(np.int32)
    diagonals = np.array([[[9, 1, 0], [6, 5, 8], [1, 2, 3], [0, 4, 5]]]).astype(np.int32)
    input_dict = {"input": input_tensor, "diagonal": diagonals[0], "name": "diag8", "k": (np.int32(-1), np.int32(2)), "align": "LEFT_RIGHT"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]).astype(np.int32)
    diagonals = np.array([[[9, 1, 0], [6, 5, 8], [1, 2, 3], [0, 4, 5]]]).astype(np.int32)
    input_dict = {"input": input_tensor, "diagonal": diagonals[0], "name": "diag9", "k": (np.int32(-1), np.int32(2)), "align": "LEFT_LEFT"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]).astype(np.int32)
    diagonals = np.array([[[9, 1, 0], [6, 5, 8], [1, 2, 3], [0, 4, 5]]]).astype(np.int32)
    input_dict = {"input": input_tensor, "diagonal": diagonals[0], "name": "diag10", "k": (np.int32(-1), np.int32(2)), "align": "RIGHT_RIGHT"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Test with float32
    input_tensor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).astype(np.float32)
    diagonal_tensor = np.array([10.0, 11.0, 12.0]).astype(np.float32)
    input_dict = {"input": input_tensor, "diagonal": diagonal_tensor, "name": "diag11", "k": np.int32(0), "align": "RIGHT_LEFT"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.set_diag"] = tf_linalg_set_diag_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.set_diag' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.set_diag'.")

check_valid('tf.linalg.set_diag', generated_inputs['tf.linalg.set_diag'], lib="tf", suffix=0)
