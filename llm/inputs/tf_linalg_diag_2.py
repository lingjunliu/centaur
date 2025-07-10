
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_diag_inputs():
    list_of_inputs = []

    # Input 1: Basic diagonal
    diagonal = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {"diagonal": diagonal, "name": "diag1", "k": (0, 0), "num_rows": -1, "num_cols": -1, "padding_value": np.array(0, dtype=np.int32), "align": "RIGHT_LEFT"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Superdiagonal
    diagonal = np.array([1, 2], dtype=np.int32)
    input_dict = {"diagonal": diagonal, "name": "diag2", "k": (1, 1), "num_rows": -1, "num_cols": -1, "padding_value": np.array(0, dtype=np.int32), "align": "RIGHT_LEFT"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Subdiagonal
    diagonal = np.array([1, 2], dtype=np.int32)
    input_dict = {"diagonal": diagonal, "name": "diag3", "k": (-1, -1), "num_rows": -1, "num_cols": -1, "padding_value": np.array(0, dtype=np.int32), "align": "RIGHT_LEFT"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tridiagonal band
    diagonal = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    input_dict = {"diagonal": diagonal, "name": "diag4", "k": (-1, 1), "num_rows": 3, "num_cols": 3, "padding_value": np.array(0, dtype=np.int32), "align": "RIGHT_LEFT"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Rectangular matrix
    diagonal = np.array([1, 2], dtype=np.int32)
    input_dict = {"diagonal": diagonal, "name": "diag5", "k": (-1, -1), "num_rows": 3, "num_cols": 4, "padding_value": np.array(0, dtype=np.int32), "align": "RIGHT_LEFT"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different padding value
    diagonal = np.array([1, 2], dtype=np.int32)
    input_dict = {"diagonal": diagonal, "name": "diag6", "k": (-1, -1), "num_rows": 3, "num_cols": 2, "padding_value": np.array(9, dtype=np.int32), "align": "RIGHT_LEFT"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Alignment LEFT_RIGHT
    diagonal = np.array([[0, 8, 9], [1, 2, 3], [4, 5, 0]], dtype=np.int32)
    input_dict = {"diagonal": diagonal, "name": "diag7", "k": (-1, 1), "num_rows": 3, "num_cols": 3, "padding_value": np.array(0, dtype=np.int32), "align": "LEFT_RIGHT"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Batch of diagonals, specify rows and cols based on diagonal length and k
    diagonal = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    input_dict = {"diagonal": diagonal, "name": "diag8", "k": (0, 0), "num_rows": 2, "num_cols": 3, "padding_value": np.array(0, dtype=np.int32), "align": "RIGHT_LEFT"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Batch of superdiagonals with specified rows and cols, k as single integer
    diagonal = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict = {"diagonal": diagonal, "name": "diag9", "k": (1, 1), "num_rows": 4, "num_cols": 4, "padding_value": np.array(0, dtype=np.int32), "align": "RIGHT_LEFT"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Float type and different k range
    diagonal = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    input_dict = {"diagonal": diagonal, "name": "diag10", "k": (-1, 1), "num_rows": 3, "num_cols": 3, "padding_value": np.array(0.0, dtype=np.float32), "align": "RIGHT_LEFT"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.diag_2"] = tf_linalg_diag_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.diag_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.diag_2'.")

check_valid('tf.linalg.diag', generated_inputs['tf.linalg.diag_2'], lib="tf", suffix=2)
