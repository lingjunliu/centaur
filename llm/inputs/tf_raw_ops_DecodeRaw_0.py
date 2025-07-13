
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_decode_raw_inputs():
    list_of_inputs = []

    # Input 1: Basic example with uint8
    bytes_data = np.array([b'\x01\x02\x03\x04', b'\x05\x06\x07\x08'], dtype=np.object_)
    out_type = tf.uint8
    little_endian = True
    name = "decode_uint8_1"
    input_dict = {"bytes": bytes_data, "out_type": out_type, "little_endian": little_endian, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32, little endian
    bytes_data = np.array([b'\x00\x00\x80\x3f', b'\x00\x00\x00\x40'], dtype=np.object_)
    out_type = tf.float32
    little_endian = True
    name = "decode_float32_1"
    input_dict = {"bytes": bytes_data, "out_type": out_type, "little_endian": little_endian, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32, big endian
    bytes_data = np.array([b'\x3f\x80\x00\x00', b'\x40\x00\x00\x00'], dtype=np.object_)
    out_type = tf.float32
    little_endian = False
    name = "decode_float32_2"
    input_dict = {"bytes": bytes_data, "out_type": out_type, "little_endian": little_endian, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int32, little endian
    bytes_data = np.array([b'\x01\x00\x00\x00', b'\xff\xff\xff\x7f'], dtype=np.object_)
    out_type = tf.int32
    little_endian = True
    name = "decode_int32_1"
    input_dict = {"bytes": bytes_data, "out_type": out_type, "little_endian": little_endian, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: int16, little endian
    bytes_data = np.array([b'\x01\x00', b'\xff\x7f'], dtype=np.object_)
    out_type = tf.int16
    little_endian = True
    name = "decode_int16_1"
    input_dict = {"bytes": bytes_data, "out_type": out_type, "little_endian": little_endian, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: uint16, big endian
    bytes_data = np.array([b'\x00\x01', b'\x7f\xff'], dtype=np.object_)
    out_type = tf.uint16
    little_endian = False
    name = "decode_uint16_1"
    input_dict = {"bytes": bytes_data, "out_type": out_type, "little_endian": little_endian, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: int8
    bytes_data = np.array([b'\x01', b'\xff'], dtype=np.object_)
    out_type = tf.int8
    little_endian = True
    name = "decode_int8_1"
    input_dict = {"bytes": bytes_data, "out_type": out_type, "little_endian": little_endian, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Multiple elements in the string tensor
    bytes_data = np.array([b'\x01\x02', b'\x03\x04', b'\x05\x06'], dtype=np.object_)
    out_type = tf.uint8
    little_endian = True
    name = "decode_uint8_2"
    input_dict = {"bytes": bytes_data, "out_type": out_type, "little_endian": little_endian, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: bfloat16, little endian
    bytes_data = np.array([b'\x00\x70', b'\x00\x40'], dtype=np.object_)
    out_type = tf.bfloat16
    little_endian = True
    name = "decode_bfloat16_1"
    input_dict = {"bytes": bytes_data, "out_type": out_type, "little_endian": little_endian, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Empty bytes, should still work (return empty tensor).
    bytes_data = np.array([b'', b''], dtype=np.object_)
    out_type = tf.float32
    little_endian = True
    name = "decode_float32_3"
    input_dict = {"bytes": bytes_data, "out_type": out_type, "little_endian": little_endian, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.DecodeRaw"] = tf_raw_ops_decode_raw_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DecodeRaw' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DecodeRaw'.")

check_valid('tf.raw_ops.DecodeRaw', generated_inputs['tf.raw_ops.DecodeRaw'], lib="tf", suffix=0)
