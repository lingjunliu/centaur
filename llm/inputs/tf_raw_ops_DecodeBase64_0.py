
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_decode_base64_inputs():
    list_of_inputs = []

    # Input 1: Basic valid case
    input1 = tf.convert_to_tensor(np.array(b"SGVsbG8gV29ybGQ=", dtype=np.object_), dtype=tf.string)
    name1 = None
    input_dict1 = {"input": input1, "name": name1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Web-safe characters
    input2 = tf.convert_to_tensor(np.array(b"_-DA", dtype=np.object_), dtype=tf.string)
    name2 = "decode_base64_2"
    input_dict2 = {"input": input2, "name": name2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: No padding
    input3 = tf.convert_to_tensor(np.array(b"SGVsbG8", dtype=np.object_), dtype=tf.string)
    name3 = None
    input_dict3 = {"input": input3, "name": name3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Empty string
    input4 = tf.convert_to_tensor(np.array(b"", dtype=np.object_), dtype=tf.string)
    name4 = "decode_base64_4"
    input_dict4 = {"input": input4, "name": name4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Multiple base64 strings
    input5 = tf.convert_to_tensor(np.array([b"SGVsbG8=", b"V29ybGQ="], dtype=np.object_), dtype=tf.string)
    name5 = None
    input_dict5 = {"input": input5, "name": name5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Longer string with web-safe chars and padding
    input6 = tf.convert_to_tensor(np.array(b"_-DA_-DA_-DA==", dtype=np.object_), dtype=tf.string)
    name6 = "decode_base64_6"
    input_dict6 = {"input": input6, "name": name6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: String with just padding
    input7 = tf.convert_to_tensor(np.array(b"====", dtype=np.object_), dtype=tf.string)
    name7 = None
    input_dict7 = {"input": input7, "name": name7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Another valid encoded string
    input8 = tf.convert_to_tensor(np.array(b"QmFzZTY0IGlzIGZ1bg==", dtype=np.object_), dtype=tf.string)
    name8 = "decode_base64_8"
    input_dict8 = {"input": input8, "name": name8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Web-safe chars only, with padding
    input9 = tf.convert_to_tensor(np.array(b"---___==", dtype=np.object_), dtype=tf.string)
    name9 = None
    input_dict9 = {"input": input9, "name": name9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: 3D Tensor of strings
    input_10 = tf.convert_to_tensor(np.array([ [b"SGVsbG8=", b"V29ybGQ="], [b"QmFzZTY0", b"---"] ], dtype=np.object_), dtype=tf.string)
    name_10 = "decode_base64_10"
    input_dict_10 = {"input": input_10, "name": name_10}
    list_of_inputs.append(copy.deepcopy(input_dict_10))

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
