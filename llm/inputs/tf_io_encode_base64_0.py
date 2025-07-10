
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_encode_base64_inputs():
    list_of_inputs = []

    # Input 1
    input_str = np.array("hello world", dtype=np.string_)
    pad = False
    name = "encode1"
    input_dict = {"input": input_str, "pad": pad, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_str = np.array("This is a test string.", dtype=np.string_)
    pad = True
    name = "encode2"
    input_dict = {"input": input_str, "pad": pad, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_str = np.array("1234567890", dtype=np.string_)
    pad = False
    name = None
    input_dict = {"input": input_str, "pad": pad, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_str = np.array("Special characters: !@#$%^&*()_+=-`~[]{}|;':\",./<>?", dtype=np.string_)
    pad = True
    name = "encode4"
    input_dict = {"input": input_str, "pad": pad, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_str = np.array("", dtype=np.string_)
    pad = False
    name = "encode5"
    input_dict = {"input": input_str, "pad": pad, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_str = np.array(["hello", "world"], dtype=np.string_)
    pad = True
    name = "encode6"
    input_dict = {"input": input_str, "pad": pad, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_str = np.array([["hello", "world"], ["test", "string"]], dtype=np.string_)
    pad = False
    name = "encode7"
    input_dict = {"input": input_str, "pad": pad, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_str = np.array("Long string to test padding: This is a long string to test the padding functionality of the base64 encoder. We need a string long enough to require padding.", dtype=np.string_)
    pad = True
    name = "encode8"
    input_dict = {"input": input_str, "pad": pad, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_str = np.array("Another test string.", dtype=np.string_)
    pad = False
    name = None
    input_dict = {"input": input_str, "pad": pad, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_str = np.array([b"binary data", b"another piece of binary data"], dtype=np.string_)
    pad = True
    name = "encode10"
    input_dict = {"input": input_str, "pad": pad, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.encode_base64"] = tf_io_encode_base64_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.encode_base64' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.encode_base64'.")

check_valid('tf.io.encode_base64', generated_inputs['tf.io.encode_base64'], lib="tf", suffix=0)
