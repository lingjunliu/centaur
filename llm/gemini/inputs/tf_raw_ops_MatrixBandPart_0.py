
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_MatrixBandPart_inputs():
    list_of_inputs = []

    # Input 1: Basic example
    input_tensor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    num_lower = np.array(1, dtype=np.int32)
    num_upper = np.array(1, dtype=np.int32)
    input_dict = {"input": input_tensor, "num_lower": num_lower, "num_upper": num_upper, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative num_lower
    input_tensor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    num_lower = np.array(-1, dtype=np.int32)
    num_upper = np.array(1, dtype=np.int32)
    input_dict = {"input": input_tensor, "num_lower": num_lower, "num_upper": num_upper, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative num_upper
    input_tensor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    num_lower = np.array(1, dtype=np.int32)
    num_upper = np.array(-1, dtype=np.int32)
    input_dict = {"input": input_tensor, "num_lower": num_lower, "num_upper": num_upper, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Both negative
    input_tensor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    num_lower = np.array(-1, dtype=np.int32)
    num_upper = np.array(-1, dtype=np.int32)
    input_dict = {"input": input_tensor, "num_lower": num_lower, "num_upper": num_upper, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger matrix
    input_tensor = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.int32)
    num_lower = np.array(2, dtype=np.int32)
    num_upper = np.array(1, dtype=np.int32)
    input_dict = {"input": input_tensor, "num_lower": num_lower, "num_upper": num_upper, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Zero values
    input_tensor = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int32)
    num_lower = np.array(1, dtype=np.int32)
    num_upper = np.array(1, dtype=np.int32)
    input_dict = {"input": input_tensor, "num_lower": num_lower, "num_upper": num_upper, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: num_lower and num_upper as int64
    input_tensor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    num_lower = np.array(1, dtype=np.int64)
    num_upper = np.array(1, dtype=np.int64)
    input_dict = {"input": input_tensor, "num_lower": num_lower, "num_upper": num_upper, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: different data type for input
    input_tensor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    num_lower = np.array(1, dtype=np.int32)
    num_upper = np.array(1, dtype=np.int32)
    input_dict = {"input": input_tensor, "num_lower": num_lower, "num_upper": num_upper, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1x1 matrix
    input_tensor = np.array([[5]], dtype=np.int32)
    num_lower = np.array(0, dtype=np.int32)
    num_upper = np.array(0, dtype=np.int32)
    input_dict = {"input": input_tensor, "num_lower": num_lower, "num_upper": num_upper, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D tensor, one row matrix
    input_tensor = np.array([[1, 2, 3]], dtype=np.int32)
    num_lower = np.array(0, dtype=np.int32)
    num_upper = np.array(2, dtype=np.int32)
    input_dict = {"input": input_tensor, "num_lower": num_lower, "num_upper": num_upper, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.MatrixBandPart"] = tf_raw_ops_MatrixBandPart_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.MatrixBandPart' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MatrixBandPart'.")

check_valid('tf.raw_ops.MatrixBandPart', generated_inputs['tf.raw_ops.MatrixBandPart'], lib="tf", suffix=0)
