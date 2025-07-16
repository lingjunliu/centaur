
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np

def tf_raw_ops_decode_raw_inputs():
    list_of_inputs = []

    # Input 1: Simple string, uint8
    bytes_val = np.array([b'hello'], dtype=np.object_)
    out_type_val = tf.uint8
    little_endian_val = True
    name_val = "decode_uint8"
    input_dict = {"bytes": bytes_val, "out_type": out_type_val, "little_endian": little_endian_val, "name": name_val}
    list_of_inputs.append(input_dict)

    # Input 2: String with multiple characters, int16, little endian
    bytes_val = np.array([b'\x01\x00\x02\x00'], dtype=np.object_)
    out_type_val = tf.int16
    little_endian_val = True
    name_val = "decode_int16_little"
    input_dict = {"bytes": bytes_val, "out_type": out_type_val, "little_endian": little_endian_val, "name": name_val}
    list_of_inputs.append(input_dict)

    # Input 3: String with multiple characters, int16, big endian
    bytes_val = np.array([b'\x00\x01\x00\x02'], dtype=np.object_)
    out_type_val = tf.int16
    little_endian_val = False
    name_val = "decode_int16_big"
    input_dict = {"bytes": bytes_val, "out_type": out_type_val, "little_endian": little_endian_val, "name": name_val}
    list_of_inputs.append(input_dict)

    # Input 4: String with multiple characters, float32, little endian
    bytes_val = np.array([b'\x00\x00\x80?'], dtype=np.object_)
    out_type_val = tf.float32
    little_endian_val = True
    name_val = "decode_float32"
    input_dict = {"bytes": bytes_val, "out_type": out_type_val, "little_endian": little_endian_val, "name": name_val}
    list_of_inputs.append(input_dict)

    # Input 5: String with multiple characters, float64, little endian
    bytes_val = np.array([b'\x00\x00\x00\x00\x00\x00\xf0?'], dtype=np.object_)
    out_type_val = tf.float64
    little_endian_val = True
    name_val = "decode_float64"
    input_dict = {"bytes": bytes_val, "out_type": out_type_val, "little_endian": little_endian_val, "name": name_val}
    list_of_inputs.append(input_dict)

    # Input 6: String with multiple elements, uint8
    bytes_val = np.array([b'ab', b'cd'], dtype=np.object_)
    out_type_val = tf.uint8
    little_endian_val = True
    name_val = "decode_uint8_multi"
    input_dict = {"bytes": bytes_val, "out_type": out_type_val, "little_endian": little_endian_val, "name": name_val}
    list_of_inputs.append(input_dict)

    # Input 7: Empty string, int32
    bytes_val = np.array([b''], dtype=np.object_)
    out_type_val = tf.int32
    little_endian_val = True
    name_val = "decode_int32_empty"
    input_dict = {"bytes": bytes_val, "out_type": out_type_val, "little_endian": little_endian_val, "name": name_val}
    list_of_inputs.append(input_dict)

    # Input 8: Longer string, int32
    bytes_val = np.array([b'\x01\x00\x00\x00\x02\x00\x00\x00'], dtype=np.object_)
    out_type_val = tf.int32
    little_endian_val = True
    name_val = "decode_int32_long"
    input_dict = {"bytes": bytes_val, "out_type": out_type_val, "little_endian": little_endian_val, "name": name_val}
    list_of_inputs.append(input_dict)

    # Input 9: String with mixed characters, complex64, little endian
    bytes_val = np.array([b'\x00\x00\x80?\x00\x00\x80?'], dtype=np.object_)
    out_type_val = tf.complex64
    little_endian_val = True
    name_val = "decode_complex64"
    input_dict = {"bytes": bytes_val, "out_type": out_type_val, "little_endian": little_endian_val, "name": name_val}
    list_of_inputs.append(input_dict)

    # Input 10: String with mixed characters, complex128, big endian
    bytes_val = np.array([b'\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\xf0?'], dtype=np.object_)
    out_type_val = tf.complex128
    little_endian_val = False
    name_val = "decode_complex128"
    input_dict = {"bytes": bytes_val, "out_type": out_type_val, "little_endian": little_endian_val, "name": name_val}
    list_of_inputs.append(input_dict)

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
