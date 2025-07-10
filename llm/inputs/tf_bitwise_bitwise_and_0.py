
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_bitwise_bitwise_and_inputs():
    list_of_inputs = []

    # Input 1: Basic int32
    x = np.array([5, 10, 15], dtype=np.int32)
    y = np.array([3, 7, 12], dtype=np.int32)
    name = "and_op_1"
    input_dict = {"x": tf.convert_to_tensor(x), "y": tf.convert_to_tensor(y), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic uint8
    x = np.array([255, 128, 0], dtype=np.uint8)
    y = np.array([127, 64, 1], dtype=np.uint8)
    name = "and_op_2"
    input_dict = {"x": tf.convert_to_tensor(x), "y": tf.convert_to_tensor(y), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multi-dimensional int16
    x = np.array([[1, 2], [3, 4]], dtype=np.int16)
    y = np.array([[2, 3], [4, 5]], dtype=np.int16)
    name = "and_op_3"
    input_dict = {"x": tf.convert_to_tensor(x), "y": tf.convert_to_tensor(y), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: All ones and zeros int64
    x = np.array([0, 1, 0, 1], dtype=np.int64)
    y = np.array([1, 0, 1, 0], dtype=np.int64)
    name = "and_op_4"
    input_dict = {"x": tf.convert_to_tensor(x), "y": tf.convert_to_tensor(y), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative values int32
    x = np.array([-5, -10, -15], dtype=np.int32)
    y = np.array([3, -7, 12], dtype=np.int32)
    name = "and_op_5"
    input_dict = {"x": tf.convert_to_tensor(x), "y": tf.convert_to_tensor(y), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different shapes uint16 (broadcasting) - shapes must be broadcastable
    x = np.array([[1, 2], [3, 4]], dtype=np.uint16)
    y = np.array([1, 3], dtype=np.uint16)
    name = "and_op_6"
    input_dict = {"x": tf.convert_to_tensor(x), "y": tf.convert_to_tensor(y), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger int32 values
    x = np.array([2147483647, -2147483648], dtype=np.int32)
    y = np.array([1, 1], dtype=np.int32)
    name = "and_op_7"
    input_dict = {"x": tf.convert_to_tensor(x), "y": tf.convert_to_tensor(y), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: uint64 max values
    x = np.array([18446744073709551615], dtype=np.uint64)
    y = np.array([1], dtype=np.uint64)
    name = "and_op_8"
    input_dict = {"x": tf.convert_to_tensor(x), "y": tf.convert_to_tensor(y), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D array int8
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int8)
    y = np.array([[[2, 3], [4, 5]], [[6, 7], [8, 9]]], dtype=np.int8)
    name = "and_op_9"
    input_dict = {"x": tf.convert_to_tensor(x), "y": tf.convert_to_tensor(y), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: uint32 with different values
    x = np.array([100, 200, 300, 400], dtype=np.uint32)
    y = np.array([50, 150, 250, 350], dtype=np.uint32)
    name = "and_op_10"
    input_dict = {"x": tf.convert_to_tensor(x), "y": tf.convert_to_tensor(y), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Scalar int32
    x = np.array(5, dtype=np.int32)
    y = np.array(3, dtype=np.int32)
    name = "and_op_11"
    input_dict = {"x": tf.convert_to_tensor(x), "y": tf.convert_to_tensor(y), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.bitwise.bitwise_and"] = tf_bitwise_bitwise_and_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.bitwise.bitwise_and' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.bitwise.bitwise_and'.")

check_valid('tf.bitwise.bitwise_and', generated_inputs['tf.bitwise.bitwise_and'], lib="tf", suffix=0)
