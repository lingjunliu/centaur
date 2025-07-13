
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_string_to_number_inputs():
    list_of_inputs = []

    # Input 1: float32
    string_tensor = tf.constant(["1.2", "3.4", "5.6"], dtype=tf.string)
    out_type = tf.float32
    name = "float32_conversion"
    input_dict = {"string_tensor": string_tensor, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: int32
    string_tensor = tf.constant(["1", "2", "3"], dtype=tf.string)
    out_type = tf.int32
    name = "int32_conversion"
    input_dict = {"string_tensor": string_tensor, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64
    string_tensor = tf.constant(["1.23456789", "9.87654321"], dtype=tf.string)
    out_type = tf.float64
    name = "float64_conversion"
    input_dict = {"string_tensor": string_tensor, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int64
    string_tensor = tf.constant(["1234567890", "9876543210"], dtype=tf.string)
    out_type = tf.int64
    name = "int64_conversion"
    input_dict = {"string_tensor": string_tensor, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: uint32
    string_tensor = tf.constant(["1000", "2000", "3000"], dtype=tf.string)
    out_type = tf.uint32
    name = "uint32_conversion"
    input_dict = {"string_tensor": string_tensor, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: uint64
    string_tensor = tf.constant(["4294967296", "8589934592"], dtype=tf.string)
    out_type = tf.uint64
    name = "uint64_conversion"
    input_dict = {"string_tensor": string_tensor, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D tensor, float32
    string_tensor = tf.constant([["1.1", "2.2"], ["3.3", "4.4"]], dtype=tf.string)
    out_type = tf.float32
    name = "2d_float32"
    input_dict = {"string_tensor": string_tensor, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.StringToNumber"] = tf_raw_ops_string_to_number_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.StringToNumber' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.StringToNumber'.")

check_valid('tf.raw_ops.StringToNumber', generated_inputs['tf.raw_ops.StringToNumber'], lib="tf", suffix=0)
