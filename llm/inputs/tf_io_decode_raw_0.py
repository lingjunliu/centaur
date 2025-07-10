
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_decode_raw_inputs():
    list_of_inputs = []

    # Input 1
    input_bytes = tf.constant(b"1234")
    out_type = np.uint8
    little_endian = True
    fixed_length = None
    name = "decode_example_1"
    input_dict = {"input_bytes": input_bytes, "out_type": out_type, "little_endian": little_endian, "fixed_length": fixed_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_bytes = tf.constant(b"abcd")
    out_type = np.int16
    little_endian = False
    fixed_length = None
    name = "decode_example_2"
    input_dict = {"input_bytes": input_bytes, "out_type": out_type, "little_endian": little_endian, "fixed_length": fixed_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_bytes = tf.constant([b"12", b"34", b"56"])
    out_type = np.uint16
    little_endian = True
    fixed_length = 4
    name = "decode_example_3"
    input_dict = {"input_bytes": input_bytes, "out_type": out_type, "little_endian": little_endian, "fixed_length": fixed_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_bytes = tf.constant([[b"1234", b"5678"], [b"9012", b"3456"]])
    out_type = np.int32
    little_endian = False
    fixed_length = 8
    name = "decode_example_4"
    input_dict = {"input_bytes": input_bytes, "out_type": out_type, "little_endian": little_endian, "fixed_length": fixed_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_bytes = tf.constant(b"abcdefgh")
    out_type = np.int64
    little_endian = True
    fixed_length = None
    name = "decode_example_5"
    input_dict = {"input_bytes": input_bytes, "out_type": out_type, "little_endian": little_endian, "fixed_length": fixed_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_bytes = tf.constant(b"87654321")
    out_type = np.uint8
    little_endian = False
    fixed_length = 8
    name = "decode_example_6"
    input_dict = {"input_bytes": input_bytes, "out_type": out_type, "little_endian": little_endian, "fixed_length": fixed_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_bytes = tf.constant([b"123456", b"789012"])
    out_type = np.int8
    little_endian = True
    fixed_length = 6
    name = "decode_example_7"
    input_dict = {"input_bytes": input_bytes, "out_type": out_type, "little_endian": little_endian, "fixed_length": fixed_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_bytes = tf.constant(b"12345678")
    out_type = np.uint32
    little_endian = True
    fixed_length = 8
    name = "decode_example_8"
    input_dict = {"input_bytes": input_bytes, "out_type": out_type, "little_endian": little_endian, "fixed_length": fixed_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_bytes = tf.constant([[b"12345678"], [b"23456789"]])
    out_type = np.int16
    little_endian = False
    fixed_length = 8
    name = "decode_example_9"
    input_dict = {"input_bytes": input_bytes, "out_type": out_type, "little_endian": little_endian, "fixed_length": fixed_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 10
    input_bytes = tf.constant([b"1234567890123456", b"5678901234567890"])
    out_type = np.int64
    little_endian = True
    fixed_length = 16
    name = "decode_example_10"
    input_dict = {"input_bytes": input_bytes, "out_type": out_type, "little_endian": little_endian, "fixed_length": fixed_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.decode_raw"] = tf_io_decode_raw_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.decode_raw' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.decode_raw'.")

check_valid('tf.io.decode_raw', generated_inputs['tf.io.decode_raw'], lib="tf", suffix=0)
