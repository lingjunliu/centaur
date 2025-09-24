
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_matrix_rank_inputs():
    list_of_inputs = []

    # Input 1
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    tol = 1e-8
    validate_args = False
    name = "matrix_rank_1"
    input_dict = {"a": a, "tol": tol, "validate_args": validate_args, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    a = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    tol = 1e-6
    validate_args = True
    name = "matrix_rank_2"
    input_dict = {"a": a, "tol": tol, "validate_args": validate_args, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    tol = 1e-8
    validate_args = False
    name = "matrix_rank_3"
    input_dict = {"a": a, "tol": tol, "validate_args": validate_args, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    a = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    tol = 1e-6
    validate_args = True
    name = "matrix_rank_4"
    input_dict = {"a": a, "tol": tol, "validate_args": validate_args, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    a = np.array([[1.0]], dtype=np.float32)
    tol = 1e-8
    validate_args = False
    name = "matrix_rank_5"
    input_dict = {"a": a, "tol": tol, "validate_args": validate_args, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    a = np.array([[0.0]], dtype=np.float64)
    tol = 1e-6
    validate_args = True
    name = "matrix_rank_6"
    input_dict = {"a": a, "tol": tol, "validate_args": validate_args, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 (batch of matrices)
    a = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    tol = 1e-8
    validate_args = False
    name = "matrix_rank_7"
    input_dict = {"a": a, "tol": tol, "validate_args": validate_args, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 (batch of matrices with rank 1)
    a = np.array([[[1.0, 1.0], [1.0, 1.0]], [[2.0, 2.0], [2.0, 2.0]]], dtype=np.float64)
    tol = 1e-6
    validate_args = True
    name = "matrix_rank_8"
    input_dict = {"a": a, "tol": tol, "validate_args": validate_args, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9 (rank deficient)
    a = np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float32)
    tol = 1e-8
    validate_args = False
    name = "matrix_rank_9"
    input_dict = {"a": a, "tol": tol, "validate_args": validate_args, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 (larger matrix)
    a = np.random.rand(10, 10).astype(np.float64)
    tol = 1e-6
    validate_args = True
    name = "matrix_rank_10"
    input_dict = {"a": a, "tol": tol, "validate_args": validate_args, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.matrix_rank"] = tf_linalg_matrix_rank_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.matrix_rank' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.matrix_rank'.")

check_valid('tf.linalg.matrix_rank', generated_inputs['tf.linalg.matrix_rank'], lib="tf", suffix=0)
