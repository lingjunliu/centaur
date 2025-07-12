
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_decode_raw_inputs():
    list_of_inputs = []

    # Input 1
    input_bytes = np.array(b"1234", dtype=np.object_)
    out_type = np.uint8
    little_endian = True
    fixed_length = None
    name = "decode_1"
    input_dict = {"input_bytes": input_bytes, "out_type": out_type, "little_endian": little_endian, "fixed_length": fixed_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_bytes = np.array([b"12", b"34"], dtype=np.object_)
    out_type = np.uint16
    little_endian = False
    fixed_length = None
    name = "decode_2"
    input_dict = {"input_bytes": input_bytes, "out_type": out_type, "little_endian": little_endian, "fixed_length": fixed_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_bytes = np.array(b"12345678", dtype=np.object_)
    out_type = np.int64
    little_endian = True
    fixed_length = None
    name = "decode_3"
    input_dict = {"input_bytes": input_bytes, "out_type": out_type, "little_endian": little_endian, "fixed_length": fixed_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_bytes = np.array([[b"1234", b"5678"], [b"9012", b"3456"]], dtype=np.object_)
    out_type = np.int16
    little_endian = True
    fixed_length = 8
    name = "decode_4"
    input_dict = {"input_bytes": input_bytes, "out_type": out_type, "little_endian": little_endian, "fixed_length": fixed_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_bytes = np.array([b'\x0a\x0b', b'\x0c\x0d'], dtype=np.object_)
    out_type = np.int16
    little_endian = False
    fixed_length = None
    name = "decode_5"
    input_dict = {"input_bytes": input_bytes, "out_type": out_type, "little_endian": little_endian, "fixed_length": fixed_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_bytes = np.array([b'\x0a\x0b\x0c\x0d'], dtype=np.object_)
    out_type = np.int32
    little_endian = True
    fixed_length = None
    name = "decode_6"
    input_dict = {"input_bytes": input_bytes, "out_type": out_type, "little_endian": little_endian, "fixed_length": fixed_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_bytes = np.array([b'\x0a\x0b\x0c\x0d'], dtype=np.object_)
    out_type = np.int32
    little_endian = False
    fixed_length = 4
    name = "decode_7"
    input_dict = {"input_bytes": input_bytes, "out_type": out_type, "little_endian": little_endian, "fixed_length": fixed_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_bytes = np.array([b"12345678", b"87654321"], dtype=np.object_)
    out_type = np.int16
    little_endian = True
    fixed_length = 8
    name = "decode_8"
    input_dict = {"input_bytes": input_bytes, "out_type": out_type, "little_endian": little_endian, "fixed_length": fixed_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_bytes = np.array([b'\x01\x02\x03\x04'], dtype=np.object_)
    out_type = np.uint16
    little_endian = True
    fixed_length = 4
    name = "decode_9"
    input_dict = {"input_bytes": input_bytes, "out_type": out_type, "little_endian": little_endian, "fixed_length": fixed_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_bytes = np.array([b'\x01\x02\x03\x04'], dtype=np.object_)
    out_type = np.uint16
    little_endian = False
    fixed_length = 4
    name = "decode_10"
    input_dict = {"input_bytes": input_bytes, "out_type": out_type, "little_endian": little_endian, "fixed_length": fixed_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_bytes = np.array(b"", dtype=np.object_)
    out_type = np.uint8
    little_endian = True
    fixed_length = None
    name = "decode_11"
    input_dict = {"input_bytes": input_bytes, "out_type": out_type, "little_endian": little_endian, "fixed_length": fixed_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    input_bytes = np.array([[b"1", b"2345678"]], dtype=np.object_)
    out_type = np.uint8
    little_endian = True
    fixed_length = 8
    name = "decode_12"
    input_dict = {"input_bytes": input_bytes, "out_type": out_type, "little_endian": little_endian, "fixed_length": fixed_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 13
    input_bytes = np.array(b"123", dtype=np.object_)
    out_type = np.uint8
    little_endian = True
    fixed_length = None
    name = "decode_13"
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
