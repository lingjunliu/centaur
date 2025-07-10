
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_cast_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array([1.8, 2.2], dtype=np.float32)
    dtype = tf.int32
    name = "cast1"
    input_dict = {"x": tf.convert_to_tensor(x), "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array([-1, 0, 1], dtype=np.int32)
    dtype = tf.float32
    name = "cast2"
    input_dict = {"x": tf.convert_to_tensor(x), "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.array([[1, 2], [3, 4]], dtype=np.int64)
    dtype = tf.float64
    name = "cast3"
    input_dict = {"x": tf.convert_to_tensor(x), "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    dtype = tf.int8
    name = "cast4"
    input_dict = {"x": tf.convert_to_tensor(x), "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.array([1, 2, 3], dtype=np.uint8)
    dtype = tf.float16
    name = "cast5"
    input_dict = {"x": tf.convert_to_tensor(x), "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = np.array([1.0 + 1j, 2.0 + 2j], dtype=np.complex64)
    dtype = tf.float32
    name = "cast6"
    input_dict = {"x": tf.convert_to_tensor(x), "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = np.array([1.0, 2.0], dtype=np.float32)
    dtype = tf.complex64
    name = "cast7"
    input_dict = {"x": tf.convert_to_tensor(x), "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = np.array([1, 2, 3, 4], dtype=np.int16)
    dtype = tf.int64
    name = "cast8"
    input_dict = {"x": tf.convert_to_tensor(x), "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    x = np.array([1.5, -2.5, 3.5], dtype=np.float32)
    dtype = tf.int32
    name = "cast9"
    input_dict = {"x": tf.convert_to_tensor(x), "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    dtype = tf.float64
    name = "cast10"
    input_dict = {"x": tf.convert_to_tensor(x), "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.cast"] = tf_cast_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.cast' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.cast'.")

check_valid('tf.cast', generated_inputs['tf.cast'], lib="tf", suffix=0)
