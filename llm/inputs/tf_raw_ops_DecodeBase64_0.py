
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_decode_base64_inputs():
    list_of_inputs = []

    # Input 1: Basic valid case
    input_str = tf.constant(np.array("SGVsbG8gV29ybGQh", dtype=np.string_))
    input_dict = {"input": input_str, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Empty string
    input_str = tf.constant(np.array("", dtype=np.string_))
    input_dict = {"input": input_str, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: String with padding
    input_str = tf.constant(np.array("SGVsbG8=", dtype=np.string_))
    input_dict = {"input": input_str, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multiple strings
    input_str = tf.constant(np.array(["SGVsbG8gV29ybGQh", "SGVsbG8gQmFzZTY0"], dtype=np.string_))
    input_dict = {"input": input_str, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: String with web-safe characters
    input_str = tf.constant(np.array("_-", dtype=np.string_))
    input_dict = {"input": input_str, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Another web safe string
    input_str = tf.constant(np.array("YWJjZA--", dtype=np.string_))
    input_dict = {"input": input_str, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Long string
    long_string = "SGVsbG8gV29ybGQhSGVsbG8gV29ybGQhSGVsbG8gV29ybGQhSGVsbG8gV29ybGQhSGVsbG8gV29ybGQh"
    input_str = tf.constant(np.array(long_string, dtype=np.string_))
    input_dict = {"input": input_str, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: String that represents numbers
    input_str = tf.constant(np.array("MTIzNDU=", dtype=np.string_))
    input_dict = {"input": input_str, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: String with non-ASCII
    input_str = tf.constant(np.array("w6TDtsOf0J/QtdGC0YDQvtCy0LDQu9C+0L3QvdC+0Lkg", dtype=np.string_))
    input_dict = {"input": input_str, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10:  A string with special characters
    input_str = tf.constant(np.array("IV4kXiomQCoh", dtype=np.string_))
    input_dict = {"input": input_str, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.DecodeBase64"] = tf_raw_ops_decode_base64_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DecodeBase64' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DecodeBase64'.")

check_valid('tf.raw_ops.DecodeBase64', generated_inputs['tf.raw_ops.DecodeBase64'], lib="tf", suffix=0)
