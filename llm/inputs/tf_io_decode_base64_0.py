
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_decode_base64_inputs():
    list_of_inputs = []

    # Input 1: Simple string
    input_str = tf.constant("SGVsbG8gV29ybGQ=")
    input_dict = {"input": input_str, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Empty string
    input_str = tf.constant("")
    input_dict = {"input": input_str, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: String with padding
    input_str = tf.constant("SGVsbG8gUGFkZGVkIQ==")
    input_dict = {"input": input_str, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: String with no padding but valid length
    input_str = tf.constant("SGVsbG8gTm9QYWRkaW5n") # Needs to be a multiple of 4
    input_dict = {"input": tf.constant("SGVsbG8gTm9QYWRkaW5n"), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Longer string
    input_str = tf.constant("VGhpcyBpcyBhIGxvbmcgYmFzZTY0IGVuY29kZWQgc3RyaW5nLiBJdCBjb250YWlucyBtb3JlIHRoYW4gdXN1YWwgbGV0dGVycy4=")
    input_dict = {"input": input_str, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: String with numbers and special characters
    input_str = tf.constant("MTIzNDU2Nzg5MCFAIyQlXiYqKCk=")
    input_dict = {"input": input_str, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Multidimensional tensor of strings
    input_str = tf.constant([["SGVsbG8=", "V29ybGQ="], ["Rm9v", "QmFy"]])
    input_dict = {"input": input_str, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Another string
    input_str = tf.constant("dGVzdGluZyBzdHJpbmc=")
    input_dict = {"input": input_str, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Web-safe with mixed casing and padding, correctly padded
    input_str = tf.constant("TmV3X3N0cmluZw==")
    input_dict = {"input": input_str, "name": "new_test"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Valid web-safe string
    input_str = tf.constant("YWJjZA")
    input_dict = {"input": tf.constant("YWJjZA"), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.decode_base64"] = tf_io_decode_base64_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.decode_base64' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.decode_base64'.")

check_valid('tf.io.decode_base64', generated_inputs['tf.io.decode_base64'], lib="tf", suffix=0)
