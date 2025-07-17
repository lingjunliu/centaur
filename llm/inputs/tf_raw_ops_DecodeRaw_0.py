
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_decode_raw_inputs():
    list_of_inputs = []

    # Input 1
    bytes_val = np.array([b'\x00\x00\x80?'], dtype=np.dtype('S4'))
    out_type_val = tf.float32
    little_endian_val = True
    name_val = None

    input_dict = {
        "bytes": bytes_val,
        "out_type": out_type_val,
        "little_endian": little_endian_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    bytes_val = np.array([b'\x00\x00\x00\x00\x00\x00\xf0?'], dtype=np.dtype('S8'))
    out_type_val = tf.float64
    little_endian_val = True
    name_val = "test_decode_raw"

    input_dict = {
        "bytes": bytes_val,
        "out_type": out_type_val,
        "little_endian": little_endian_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    bytes_val = np.array([b'\x01\x02\x03\x04'], dtype=np.dtype('S4'))
    out_type_val = tf.int32
    little_endian_val = False
    name_val = None

    input_dict = {
        "bytes": bytes_val,
        "out_type": out_type_val,
        "little_endian": little_endian_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    bytes_val = np.array([b'\x01\x02'], dtype=np.dtype('S2'))
    out_type_val = tf.int16
    little_endian_val = True
    name_val = None

    input_dict = {
        "bytes": bytes_val,
        "out_type": out_type_val,
        "little_endian": little_endian_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    bytes_val = np.array([b'\xff'], dtype=np.dtype('S1'))
    out_type_val = tf.int8
    little_endian_val = True
    name_val = None

    input_dict = {
        "bytes": bytes_val,
        "out_type": out_type_val,
        "little_endian": little_endian_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    bytes_val = np.array([b'\x01\x00'], dtype=np.dtype('S2'))
    out_type_val = tf.uint16
    little_endian_val = True
    name_val = None

    input_dict = {
        "bytes": bytes_val,
        "out_type": out_type_val,
        "little_endian": little_endian_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    bytes_val = np.array([b'\x01'], dtype=np.dtype('S1'))
    out_type_val = tf.uint8
    little_endian_val = True
    name_val = None

    input_dict = {
        "bytes": bytes_val,
        "out_type": out_type_val,
        "little_endian": little_endian_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    bytes_val = np.array([b'\x00\x00\x00\x00\x00\x00\x00\x00'], dtype=np.dtype('S8'))
    out_type_val = tf.int64
    little_endian_val = True
    name_val = None

    input_dict = {
        "bytes": bytes_val,
        "out_type": out_type_val,
        "little_endian": little_endian_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    bytes_val = np.array([b'\x00\x00\x80\x3f\x00\x00\x80\xbf'], dtype=np.dtype('S8'))
    out_type_val = tf.complex64
    little_endian_val = True
    name_val = None

    input_dict = {
        "bytes": bytes_val,
        "out_type": out_type_val,
        "little_endian": little_endian_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    bytes_val = np.array([b'\x01'], dtype=np.dtype('S1'))
    out_type_val = tf.bool
    little_endian_val = True
    name_val = None

    input_dict = {
        "bytes": bytes_val,
        "out_type": out_type_val,
        "little_endian": little_endian_val,
        "name": name_val
    }
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
