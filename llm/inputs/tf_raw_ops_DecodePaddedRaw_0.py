
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_DecodePaddedRaw_inputs():
    list_of_inputs = []

    # Input 1
    input_bytes = np.array([b'\x01\x02\x03\x04'], dtype=np.string_)
    fixed_length = np.array(4, dtype=np.int32)
    out_type = tf.int32
    little_endian = True
    name = "decode_1"
    input_dict = {"input_bytes": input_bytes, "fixed_length": fixed_length, "out_type": out_type, "little_endian": little_endian, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_bytes = np.array([b'\x01\x02\x03\x04\x05\x06\x07\x08'], dtype=np.string_)
    fixed_length = np.array(8, dtype=np.int32)
    out_type = tf.int64
    little_endian = False
    name = "decode_2"
    input_dict = {"input_bytes": input_bytes, "fixed_length": fixed_length, "out_type": out_type, "little_endian": little_endian, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_bytes = np.array([b'\x00\x00\x80\x3f'], dtype=np.string_)
    fixed_length = np.array(4, dtype=np.int32)
    out_type = tf.float32
    little_endian = True
    name = "decode_3"
    input_dict = {"input_bytes": input_bytes, "fixed_length": fixed_length, "out_type": out_type, "little_endian": little_endian, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_bytes = np.array([b'\x01\x00'], dtype=np.string_)
    fixed_length = np.array(2, dtype=np.int32)
    out_type = tf.int16
    little_endian = True
    name = "decode_4"
    input_dict = {"input_bytes": input_bytes, "fixed_length": fixed_length, "out_type": out_type, "little_endian": little_endian, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_bytes = np.array([b'\x01\x02', b'\x03\x04'], dtype=np.string_)
    fixed_length = np.array(1, dtype=np.int32)
    out_type = tf.int8
    little_endian = True
    name = "decode_5"
    input_dict = {"input_bytes": input_bytes, "fixed_length": fixed_length, "out_type": out_type, "little_endian": little_endian, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_bytes = np.array([b'\x01', b'\x02', b'\x03', b'\x04', b'\x05', b'\x06', b'\x07', b'\x08'], dtype=np.string_)
    fixed_length = np.array(1, dtype=np.int32)
    out_type = tf.uint8
    little_endian = True
    name = "decode_6"
    input_dict = {"input_bytes": input_bytes, "fixed_length": fixed_length, "out_type": out_type, "little_endian": little_endian, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7
    input_bytes = np.array([b'\x01\x02\x03\x04'], dtype=np.string_)
    fixed_length = np.array(4, dtype=np.int32)
    out_type = tf.int32
    little_endian = True
    name = "decode_7"
    input_dict = {"input_bytes": input_bytes, "fixed_length": fixed_length, "out_type": out_type, "little_endian": little_endian, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_bytes = np.array([b'\x00\x00\x00\x00\x00\x00\x00\x00'], dtype=np.string_)
    fixed_length = np.array(8, dtype=np.int32)
    out_type = tf.float64
    little_endian = True
    name = "decode_8"
    input_dict = {"input_bytes": input_bytes, "fixed_length": fixed_length, "out_type": out_type, "little_endian": little_endian, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_bytes = np.array([b'\x00\x00\x00\x00'], dtype=np.string_)
    fixed_length = np.array(4, dtype=np.int32)
    out_type = tf.int32
    little_endian = True
    name = "decode_9"
    input_dict = {"input_bytes": input_bytes, "fixed_length": fixed_length, "out_type": out_type, "little_endian": little_endian, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_bytes = np.array([b'\x01\x02\x03\x04\x05\x06'], dtype=np.string_)
    fixed_length = np.array(6, dtype=np.int32)
    out_type = tf.uint8
    little_endian = True
    name = "decode_10"
    input_dict = {"input_bytes": input_bytes, "fixed_length": fixed_length, "out_type": out_type, "little_endian": little_endian, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.DecodePaddedRaw"] = tf_raw_ops_DecodePaddedRaw_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DecodePaddedRaw' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DecodePaddedRaw'.")

check_valid('tf.raw_ops.DecodePaddedRaw', generated_inputs['tf.raw_ops.DecodePaddedRaw'], lib="tf", suffix=0)
