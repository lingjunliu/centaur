
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_decode_padded_raw_inputs():
    list_of_inputs = []

    # Input 1
    input_bytes = tf.constant([b'\x01\x02\x03\x04'])
    fixed_length = tf.constant(4, dtype=tf.int32)
    out_type = tf.uint8
    little_endian = True
    name = None
    input_dict = {"input_bytes": input_bytes, "fixed_length": fixed_length, "out_type": out_type, "little_endian": little_endian, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_bytes = tf.constant([b'\x01\x02\x03\x04\x05\x06\x07\x08'])
    fixed_length = tf.constant(8, dtype=tf.int32)
    out_type = tf.uint8
    little_endian = False
    name = "test2"
    input_dict = {"input_bytes": input_bytes, "fixed_length": fixed_length, "out_type": out_type, "little_endian": little_endian, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_bytes = tf.constant([b'\x01\x00\x00\x00\x02\x00\x00\x00'])
    fixed_length = tf.constant(8, dtype=tf.int32)
    out_type = tf.int32
    little_endian = True
    name = None
    input_dict = {"input_bytes": input_bytes, "fixed_length": fixed_length, "out_type": out_type, "little_endian": little_endian, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_bytes = tf.constant([b'\x00\x00\x00\x01\x00\x00\x00\x02'])
    fixed_length = tf.constant(8, dtype=tf.int32)
    out_type = tf.int32
    little_endian = False
    name = None
    input_dict = {"input_bytes": input_bytes, "fixed_length": fixed_length, "out_type": out_type, "little_endian": little_endian, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_bytes = tf.constant([b'\x01\x00'])
    fixed_length = tf.constant(2, dtype=tf.int32)
    out_type = tf.uint16
    little_endian = True
    name = None
    input_dict = {"input_bytes": input_bytes, "fixed_length": fixed_length, "out_type": out_type, "little_endian": little_endian, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_bytes = tf.constant([b'\x00\x01'])
    fixed_length = tf.constant(2, dtype=tf.int32)
    out_type = tf.uint16
    little_endian = False
    name = None
    input_dict = {"input_bytes": input_bytes, "fixed_length": fixed_length, "out_type": out_type, "little_endian": little_endian, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_bytes = tf.constant([b'\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10'])
    fixed_length = tf.constant(16, dtype=tf.int32)
    out_type = tf.uint8
    little_endian = True
    name = None
    input_dict = {"input_bytes": input_bytes, "fixed_length": fixed_length, "out_type": out_type, "little_endian": little_endian, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_bytes = tf.constant([b'\x00\x00\x80\x3f'])
    fixed_length = tf.constant(4, dtype=tf.int32)
    out_type = tf.float32
    little_endian = True
    name = None
    input_dict = {"input_bytes": input_bytes, "fixed_length": fixed_length, "out_type": out_type, "little_endian": little_endian, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_bytes = tf.constant([b'\x3f\x80\x00\x00'])
    fixed_length = tf.constant(4, dtype=tf.int32)
    out_type = tf.float32
    little_endian = False
    name = None
    input_dict = {"input_bytes": input_bytes, "fixed_length": fixed_length, "out_type": out_type, "little_endian": little_endian, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 10
    input_bytes = tf.constant([b'\x00\x00\x00\x00\x00\x00\xf0\x3f'])
    fixed_length = tf.constant(8, dtype=tf.int32)
    out_type = tf.float64
    little_endian = True
    name = None
    input_dict = {"input_bytes": input_bytes, "fixed_length": fixed_length, "out_type": out_type, "little_endian": little_endian, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.DecodePaddedRaw"] = tf_raw_ops_decode_padded_raw_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DecodePaddedRaw' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DecodePaddedRaw'.")

check_valid('tf.raw_ops.DecodePaddedRaw', generated_inputs['tf.raw_ops.DecodePaddedRaw'], lib="tf", suffix=0)
