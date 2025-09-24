
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_eigh_inputs():
    list_of_inputs = []

    # Input 1: Basic 2x2 matrix
    tensor = np.array([[2, 1], [1, 3]], dtype=np.float32)
    name = "eigh_1"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Batch of 2x2 matrices
    tensor = np.array([[[2, 1], [1, 3]], [[4, 2], [2, 5]]], dtype=np.float32)
    name = "eigh_2"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3x3 matrix
    tensor = np.array([[4, 1, 2], [1, 5, 3], [2, 3, 6]], dtype=np.float32)
    name = "eigh_3"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Batch of 3x3 matrices
    tensor = np.array([[[4, 1, 2], [1, 5, 3], [2, 3, 6]], [[7, 2, 1], [2, 8, 4], [1, 4, 9]]], dtype=np.float32)
    name = "eigh_4"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger matrix (4x4)
    tensor = np.array([[1, 2, 3, 4], [2, 5, 6, 7], [3, 6, 8, 9], [4, 7, 9, 10]], dtype=np.float32)
    name = "eigh_5"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Matrix with negative values
    tensor = np.array([[1, -2], [-2, 3]], dtype=np.float32)
    name = "eigh_6"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Batch of matrices with negative values
    tensor = np.array([[[1, -2], [-2, 3]], [[-1, 2], [2, -3]]], dtype=np.float32)
    name = "eigh_7"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8: A more complex 3x3 matrix
    tensor = np.array([[10, -1, 0], [-1, 8, -2], [0, -2, 6]], dtype=np.float32)
    name = "eigh_8"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Different data type (float64)
    tensor = np.array([[2, 1], [1, 3]], dtype=np.float64)
    name = "eigh_9"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5x5 matrix
    tensor = np.array([[1, 0, 0, 0, 0],
                       [0, 2, 0, 0, 0],
                       [0, 0, 3, 0, 0],
                       [0, 0, 0, 4, 0],
                       [0, 0, 0, 0, 5]], dtype=np.float32)
    name = "eigh_10"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.eigh"] = tf_linalg_eigh_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.eigh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.eigh'.")

check_valid('tf.linalg.eigh', generated_inputs['tf.linalg.eigh'], lib="tf", suffix=0)
