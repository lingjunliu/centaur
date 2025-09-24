
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_lstsq_inputs():
    list_of_inputs = []

    # Input 1
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    rhs = np.array([[1.0], [2.0]], dtype=np.float32)
    l2_regularizer = 0.0
    fast = True
    name = "lstsq_1"
    input_dict = {"matrix": matrix, "rhs": rhs, "l2_regularizer": l2_regularizer, "fast": fast, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    matrix = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    rhs = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    l2_regularizer = 0.1
    fast = False
    name = "lstsq_2"
    input_dict = {"matrix": matrix, "rhs": rhs, "l2_regularizer": l2_regularizer, "fast": fast, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    rhs = np.array([[7.0, 8.0], [9.0, 10.0]], dtype=np.float32)
    l2_regularizer = 0.01
    fast = True
    name = "lstsq_3"
    input_dict = {"matrix": matrix, "rhs": rhs, "l2_regularizer": l2_regularizer, "fast": fast, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    rhs = np.array([[1.0], [2.0]], dtype=np.float64)
    l2_regularizer = 0.0
    fast = True
    name = "lstsq_4"
    input_dict = {"matrix": matrix, "rhs": rhs, "l2_regularizer": l2_regularizer, "fast": fast, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    rhs = np.array([[7.0], [9.0]], dtype=np.float32)
    l2_regularizer = 0.5
    fast = False
    name = "lstsq_5"
    input_dict = {"matrix": matrix, "rhs": rhs, "l2_regularizer": l2_regularizer, "fast": fast, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    rhs = np.array([[1.0, 5.0], [2.0, 6.0]], dtype=np.float32)
    l2_regularizer = 0.0
    fast = True
    name = "lstsq_6"
    input_dict = {"matrix": matrix, "rhs": rhs, "l2_regularizer": l2_regularizer, "fast": fast, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    matrix = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    rhs = np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float32)
    l2_regularizer = 0.0
    fast = True
    name = "lstsq_7"
    input_dict = {"matrix": matrix, "rhs": rhs, "l2_regularizer": l2_regularizer, "fast": fast, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8
    matrix = np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float32)
    rhs = np.array([[1.0], [1.0]], dtype=np.float32)
    l2_regularizer = 1.0
    fast = True
    name = "lstsq_8"
    input_dict = {"matrix": matrix, "rhs": rhs, "l2_regularizer": l2_regularizer, "fast": fast, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    rhs = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    l2_regularizer = 0.0
    fast = True
    name = "lstsq_9"
    input_dict = {"matrix": matrix, "rhs": rhs, "l2_regularizer": l2_regularizer, "fast": fast, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    rhs = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    l2_regularizer = 0.01
    fast = False
    name = "lstsq_10"
    input_dict = {"matrix": matrix, "rhs": rhs, "l2_regularizer": l2_regularizer, "fast": fast, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.lstsq"] = tf_linalg_lstsq_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.lstsq' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.lstsq'.")

check_valid('tf.linalg.lstsq', generated_inputs['tf.linalg.lstsq'], lib="tf", suffix=0)
