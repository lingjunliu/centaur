
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_bitcast_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([1, 2, 3], dtype=np.int32)
    dtype = tf.float32
    name = "bitcast_1"
    input_dict = {"input": input_tensor, "type": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    dtype = tf.int32
    name = "bitcast_2"
    input_dict = {"input": input_tensor, "type": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.int32)
    dtype = tf.uint8
    name = "bitcast_3"
    input_dict = {"input": input_tensor, "type": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint8)
    dtype = tf.int32
    name = "bitcast_4"
    input_dict = {"input": input_tensor, "type": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([1.0 + 1j, 2.0 + 2j], dtype=np.complex64)
    dtype = tf.float32
    name = "bitcast_5"
    input_dict = {"input": input_tensor, "type": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([1, 2], dtype=np.int64)
    dtype = tf.int32
    name = "bitcast_6"
    input_dict = {"input": input_tensor, "type": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([1, 2, 3, 4], dtype=np.int16)
    dtype = tf.uint8
    name = "bitcast_7"
    input_dict = {"input": input_tensor, "type": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([1.0, 2.0], dtype=np.float64)
    dtype = tf.int64
    name = "bitcast_8"
    input_dict = {"input": input_tensor, "type": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([1, 2, 3, 4], dtype=np.uint32)
    dtype = tf.uint8
    name = "bitcast_9"
    input_dict = {"input": input_tensor, "type": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([1], dtype=np.int8)
    dtype = tf.uint8
    name = "bitcast_10"
    input_dict = {"input": input_tensor, "type": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_raw_ops_bitcast_inputs()
generated_inputs["tf.raw_ops.Bitcast"] = []
for input_dict in inputs:
    generated_inputs["tf.raw_ops.Bitcast"].append({"kwargs": input_dict})

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Bitcast' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Bitcast'.")

check_valid('tf.raw_ops.Bitcast', generated_inputs['tf.raw_ops.Bitcast'], lib="tf", suffix=0)
