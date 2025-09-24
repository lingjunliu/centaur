
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import gzip
import zlib
import copy

def tf_io_decode_compressed_inputs():
    list_of_inputs = []

    # Input 1: Empty string, no compression
    bytes_data = np.array([b""], dtype=np.object_)
    compression_type = ""
    name = "decode_empty"
    input_dict = {"bytes": bytes_data, "compression_type": compression_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Simple string, no compression
    bytes_data = np.array([b"hello world"], dtype=np.object_)
    compression_type = ""
    name = "decode_simple"
    input_dict = {"bytes": bytes_data, "compression_type": compression_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: ZLIB compressed string
    data = b"hello world"
    compressed_data = zlib.compress(data)
    bytes_data = np.array([compressed_data], dtype=np.object_)
    compression_type = "ZLIB"
    name = "decode_zlib"
    input_dict = {"bytes": bytes_data, "compression_type": compression_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: GZIP compressed string
    data = b"hello world"
    compressed_data = gzip.compress(data)
    bytes_data = np.array([compressed_data], dtype=np.object_)
    compression_type = "GZIP"
    name = "decode_gzip"
    input_dict = {"bytes": bytes_data, "compression_type": compression_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multiple strings, no compression
    bytes_data = np.array([b"hello", b"world", b"!"], dtype=np.object_)
    compression_type = ""
    name = "decode_multiple"
    input_dict = {"bytes": bytes_data, "compression_type": compression_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multiple strings, ZLIB compression
    data1 = b"hello"
    data2 = b"world"
    data3 = b"!"
    compressed_data1 = zlib.compress(data1)
    compressed_data2 = zlib.compress(data2)
    compressed_data3 = zlib.compress(data3)
    bytes_data = np.array([compressed_data1, compressed_data2, compressed_data3], dtype=np.object_)
    compression_type = "ZLIB"
    name = "decode_multiple_zlib"
    input_dict = {"bytes": bytes_data, "compression_type": compression_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Multiple strings, GZIP compression
    data1 = b"hello"
    data2 = b"world"
    data3 = b"!"
    compressed_data1 = gzip.compress(data1)
    compressed_data2 = gzip.compress(data2)
    compressed_data3 = gzip.compress(data3)
    bytes_data = np.array([compressed_data1, compressed_data2, compressed_data3], dtype=np.object_)
    compression_type = "GZIP"
    name = "decode_multiple_gzip"
    input_dict = {"bytes": bytes_data, "compression_type": compression_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Longer string, ZLIB
    data = b"This is a longer string to test ZLIB compression."
    compressed_data = zlib.compress(data)
    bytes_data = np.array([compressed_data], dtype=np.object_)
    compression_type = "ZLIB"
    name = "decode_long_zlib"
    input_dict = {"bytes": bytes_data, "compression_type": compression_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Longer string, GZIP
    data = b"This is a longer string to test GZIP compression."
    compressed_data = gzip.compress(data)
    bytes_data = np.array([compressed_data], dtype=np.object_)
    compression_type = "GZIP"
    name = "decode_long_gzip"
    input_dict = {"bytes": bytes_data, "compression_type": compression_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: String with special characters, ZLIB
    data = b"String with special characters: !@#$%^&*()"
    compressed_data = zlib.compress(data)
    bytes_data = np.array([compressed_data], dtype=np.object_)
    compression_type = "ZLIB"
    name = "decode_special_zlib"
    input_dict = {"bytes": bytes_data, "compression_type": compression_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.decode_compressed"] = tf_io_decode_compressed_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.decode_compressed' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.decode_compressed'.")

check_valid('tf.io.decode_compressed', generated_inputs['tf.io.decode_compressed'], lib="tf", suffix=0)
