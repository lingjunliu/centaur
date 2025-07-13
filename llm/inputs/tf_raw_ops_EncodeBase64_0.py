
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_encode_base64_inputs():
    list_of_inputs = []

    # Input 1
    input_val = tf.constant(np.array([b"hello"]).astype("S"))
    pad_val = False
    name_val = None
    input_dict = {"input": input_val, "pad": pad_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_val = tf.constant(np.array([b"this is a longer string"]).astype("S"))
    pad_val = True
    name_val = "encode1"
    input_dict = {"input": input_val, "pad": pad_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_val = tf.constant(np.array([b"string1", b"string2", b"string3"]).astype("S"))
    pad_val = False
    name_val = "encode2"
    input_dict = {"input": input_val, "pad": pad_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_val = tf.constant(np.array([b"string1", b"string2", b""]).astype("S"))
    pad_val = True
    name_val = None
    input_dict = {"input": input_val, "pad": pad_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_val = tf.constant(np.array([b""]).astype("S"))
    pad_val = False
    name_val = "encode3"
    input_dict = {"input": input_val, "pad": pad_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_val = tf.constant(np.array([b"1234567890"]).astype("S"))
    pad_val = True
    name_val = None
    input_dict = {"input": input_val, "pad": pad_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_val = tf.constant(np.array([b"", b"", b""]).astype("S"))
    pad_val = False
    name_val = None
    input_dict = {"input": input_val, "pad": pad_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    input_val = tf.constant(np.array([b"abcdefghijklmnopqrstuvwxyz"]).astype("S"))
    pad_val = True
    name_val = "encode4"
    input_dict = {"input": input_val, "pad": pad_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_val = tf.constant(np.array([b"!@#$%^&*()"]).astype("S"))
    pad_val = False
    name_val = None
    input_dict = {"input": input_val, "pad": pad_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_val = tf.constant(np.array([b"a", b"b", b"c", b"d", b"e", b"f", b"g"]).astype("S"))
    pad_val = True
    name_val = "encode5"
    input_dict = {"input": input_val, "pad": pad_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.EncodeBase64"] = tf_raw_ops_encode_base64_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.EncodeBase64' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.EncodeBase64'.")

check_valid('tf.raw_ops.EncodeBase64', generated_inputs['tf.raw_ops.EncodeBase64'], lib="tf", suffix=0)
