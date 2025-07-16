
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_decode_base64_inputs():
    list_of_inputs = []

    # Input 1: Basic valid base64 string
    input_str = tf.constant(np.array("SGVsbG8gV29ybGQ=").astype(np.string_))
    input_dict = {"input": input_str, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: String with padding
    input_str = tf.constant(np.array("SGVsbG8gV29ybGQA").astype(np.string_))
    input_dict = {"input": input_str, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Web-safe base64
    input_str = tf.constant(np.array("_-").astype(np.string_))
    input_dict = {"input": input_str, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty string
    input_str = tf.constant(np.array("").astype(np.string_))
    input_dict = {"input": input_str, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Longer string
    input_str = tf.constant(np.array("VGhpcyBpcyBhIGxvbmcgYmFzZTY0IGVuY29kZWQgc3RyaW5nLg==").astype(np.string_))
    input_dict = {"input": input_str, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: String with special characters
    input_str = tf.constant(np.array("IUAjJCVeJiooKWB+PT1bXXt9fDpcIjsnLC4vPD4/Pw==").astype(np.string_))
    input_dict = {"input": input_str, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Multi-dimensional tensor
    input_str = tf.constant(np.array([["SGVsbG8=","V29ybGQ="],["SGVsbG8=","V29ybGQ="]]).astype(np.string_))
    input_dict = {"input": input_str, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1-dimensional tensor
    input_str = tf.constant(np.array(["SGVsbG8=", "V29ybGQ="]).astype(np.string_))
    input_dict = {"input": input_str, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Another web-safe string
    input_str = tf.constant(np.array("YS1i").astype(np.string_))
    input_dict = {"input": input_str, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Base64 for a single character
    input_str = tf.constant(np.array("YQ==").astype(np.string_))
    input_dict = {"input": input_str, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Adding Name
    input_str = tf.constant(np.array("SGVsbG8gV29ybGQ=").astype(np.string_))
    input_dict = {"input": input_str, "name": "my_decode"}
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
