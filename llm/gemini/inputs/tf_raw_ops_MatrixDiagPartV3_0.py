
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_MatrixDiagPartV3_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k_tensor = np.array(0, dtype=np.int32)
    padding_value_tensor = np.array(0, dtype=input_tensor.dtype)
    align_str = "RIGHT_LEFT"
    name_str = None

    input_dict = {
        "input": input_tensor,
        "k": k_tensor,
        "padding_value": padding_value_tensor,
        "align": align_str,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k_tensor = np.array(1, dtype=np.int32)
    padding_value_tensor = np.array(0, dtype=input_tensor.dtype)
    align_str = "RIGHT_LEFT"
    name_str = None

    input_dict = {
        "input": input_tensor,
        "k": k_tensor,
        "padding_value": padding_value_tensor,
        "align": align_str,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k_tensor = np.array(-1, dtype=np.int32)
    padding_value_tensor = np.array(0, dtype=input_tensor.dtype)
    align_str = "RIGHT_LEFT"
    name_str = None

    input_dict = {
        "input": input_tensor,
        "k": k_tensor,
        "padding_value": padding_value_tensor,
        "align": align_str,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k_tensor = np.array([0, 1], dtype=np.int32)
    padding_value_tensor = np.array(0, dtype=input_tensor.dtype)
    align_str = "RIGHT_LEFT"
    name_str = None

    input_dict = {
        "input": input_tensor,
        "k": k_tensor,
        "padding_value": padding_value_tensor,
        "align": align_str,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k_tensor = np.array([-1, 1], dtype=np.int32)
    padding_value_tensor = np.array(0, dtype=input_tensor.dtype)
    align_str = "RIGHT_LEFT"
    name_str = None

    input_dict = {
        "input": input_tensor,
        "k": k_tensor,
        "padding_value": padding_value_tensor,
        "align": align_str,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k_tensor = np.array([-2, -1], dtype=np.int32)
    padding_value_tensor = np.array(0, dtype=input_tensor.dtype)
    align_str = "RIGHT_LEFT"
    name_str = None

    input_dict = {
        "input": input_tensor,
        "k": k_tensor,
        "padding_value": padding_value_tensor,
        "align": align_str,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k_tensor = np.array(0, dtype=np.int32)
    padding_value_tensor = np.array(9, dtype=input_tensor.dtype)
    align_str = "RIGHT_LEFT"
    name_str = None

    input_dict = {
        "input": input_tensor,
        "k": k_tensor,
        "padding_value": padding_value_tensor,
        "align": align_str,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    k_tensor = np.array(0, dtype=np.int32)
    padding_value_tensor = np.array(0, dtype=input_tensor.dtype)
    align_str = "RIGHT_LEFT"
    name_str = None

    input_dict = {
        "input": input_tensor,
        "k": k_tensor,
        "padding_value": padding_value_tensor,
        "align": align_str,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    k_tensor = np.array([-1, 1], dtype=np.int32)
    padding_value_tensor = np.array(9, dtype=input_tensor.dtype)
    align_str = "LEFT_RIGHT"
    name_str = None

    input_dict = {
        "input": input_tensor,
        "k": k_tensor,
        "padding_value": padding_value_tensor,
        "align": align_str,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]])
    k_tensor = np.array([-1, 2], dtype=np.int32)
    padding_value_tensor = np.array(-1, dtype=input_tensor.dtype)
    align_str = "LEFT_LEFT"
    name_str = None

    input_dict = {
        "input": input_tensor,
        "k": k_tensor,
        "padding_value": padding_value_tensor,
        "align": align_str,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.MatrixDiagPartV3"] = tf_raw_ops_MatrixDiagPartV3_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.MatrixDiagPartV3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MatrixDiagPartV3'.")

check_valid('tf.raw_ops.MatrixDiagPartV3', generated_inputs['tf.raw_ops.MatrixDiagPartV3'], lib="tf", suffix=0)
