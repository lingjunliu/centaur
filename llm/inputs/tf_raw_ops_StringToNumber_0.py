
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_string_to_number_inputs():
    list_of_inputs = []

    # Input 1: float32, simple case
    string_tensor = np.array(["1.0", "2.0", "3.0"], dtype="S3")
    out_type = tf.float32
    name = "float_conversion"
    input_dict = {"string_tensor": string_tensor, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: int32
    string_tensor = np.array(["1", "2", "3"], dtype="S1")
    out_type = tf.int32
    name = "int_conversion"
    input_dict = {"string_tensor": string_tensor, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64
    string_tensor = np.array(["1.1", "2.2", "3.3"], dtype="S3")
    out_type = tf.float64
    name = "double_conversion"
    input_dict = {"string_tensor": string_tensor, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int64
    string_tensor = np.array(["-1", "2", "3"], dtype="S2")
    out_type = tf.int64
    name = "int64_conversion"
    input_dict = {"string_tensor": string_tensor, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: uint32
    string_tensor = np.array(["1", "2", "3"], dtype="S1")
    out_type = tf.uint32
    name = "uint32_conversion"
    input_dict = {"string_tensor": string_tensor, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: uint64
    string_tensor = np.array(["1", "2", "3"], dtype="S1")
    out_type = tf.uint64
    name = "uint64_conversion"
    input_dict = {"string_tensor": string_tensor, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32, with negative values
    string_tensor = np.array(["-1.0", "2.0", "-3.0"], dtype="S4")
    out_type = tf.float32
    name = "float_conversion_negative"
    input_dict = {"string_tensor": string_tensor, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D array, int32
    string_tensor = np.array([["1", "2"], ["3", "4"]], dtype="S1")
    out_type = tf.int32
    name = "int_conversion_2d"
    input_dict = {"string_tensor": string_tensor, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32, with zero
    string_tensor = np.array(["0.0", "2.0", "3.0"], dtype="S3")
    out_type = tf.float32
    name = "float_conversion_zero"
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
