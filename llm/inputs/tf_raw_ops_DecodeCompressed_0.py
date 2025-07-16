
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
import gzip
import zlib

def tf_raw_ops_decode_compressed_inputs():
    list_of_inputs = []

    # Input 1: Empty string, no compression
    bytes_data = np.array([b""], dtype=np.dtype('S'))
    compression_type = ""
    name = None
    input_dict = {"bytes": bytes_data, "compression_type": compression_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Simple ZLIB compression
    data = b"Hello, world!"
    compressed_data = zlib.compress(data)
    bytes_data = np.array([compressed_data], dtype=np.dtype('S'))
    compression_type = "ZLIB"
    name = "zlib_test"
    input_dict = {"bytes": bytes_data, "compression_type": compression_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Simple GZIP compression
    data = b"This is a test string for GZIP compression."
    compressed_data = gzip.compress(data)
    bytes_data = np.array([compressed_data], dtype=np.dtype('S'))
    compression_type = "GZIP"
    name = "gzip_test"
    input_dict = {"bytes": bytes_data, "compression_type": compression_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: No compression, multiple strings
    data1 = b"String one"
    data2 = b"String two"
    bytes_data = np.array([data1, data2], dtype=np.dtype('S'))
    compression_type = ""
    name = None
    input_dict = {"bytes": bytes_data, "compression_type": compression_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: ZLIB with multiple elements
    data1 = b"Short zlib string"
    data2 = b"Another zlib string, longer this time"
    compressed_data1 = zlib.compress(data1)
    compressed_data2 = zlib.compress(data2)
    bytes_data = np.array([compressed_data1, compressed_data2], dtype=np.dtype('S'))
    compression_type = "ZLIB"
    name = "zlib_multiple"
    input_dict = {"bytes": bytes_data, "compression_type": compression_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: GZIP with multiple elements
    data1 = b"Short gzip string"
    data2 = b"Another gzip string, longer this time"
    compressed_data1 = gzip.compress(data1)
    compressed_data2 = gzip.compress(data2)
    bytes_data = np.array([compressed_data1, compressed_data2], dtype=np.dtype('S'))
    compression_type = "GZIP"
    name = "gzip_multiple"
    input_dict = {"bytes": bytes_data, "compression_type": compression_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8: ZLIB with empty string in the array
    data1 = b""
    data2 = b"Another zlib string, longer this time"
    compressed_data1 = zlib.compress(data1)
    compressed_data2 = zlib.compress(data2)
    bytes_data = np.array([compressed_data1, compressed_data2], dtype=np.dtype('S'))
    compression_type = "ZLIB"
    name = "zlib_multiple_empty"
    input_dict = {"bytes": bytes_data, "compression_type": compression_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: GZIP with empty string in the array
    data1 = b""
    data2 = b"Another gzip string, longer this time"
    compressed_data1 = gzip.compress(data1)
    compressed_data2 = gzip.compress(data2)
    bytes_data = np.array([compressed_data1, compressed_data2], dtype=np.dtype('S'))
    compression_type = "GZIP"
    name = "gzip_multiple_empty"
    input_dict = {"bytes": bytes_data, "compression_type": compression_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Multidimensional array with no compression
    data = [b"string1", b"string2", b"string3", b"string4"]
    bytes_data = np.array(data, dtype=np.dtype('S')).reshape((2, 2))
    compression_type = ""
    name = "multidimensional_no_compression"
    input_dict = {"bytes": bytes_data, "compression_type": compression_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.DecodeCompressed"] = tf_raw_ops_decode_compressed_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DecodeCompressed' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DecodeCompressed'.")

check_valid('tf.raw_ops.DecodeCompressed', generated_inputs['tf.raw_ops.DecodeCompressed'], lib="tf", suffix=0)
