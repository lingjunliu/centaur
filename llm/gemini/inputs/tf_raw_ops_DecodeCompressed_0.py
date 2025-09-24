
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
import zlib
import gzip

def tf_raw_ops_decodecompressed_inputs():
    list_of_inputs = []

    # Input 1: Empty string, no compression
    bytes_data = np.array([b""], dtype=np.object_)
    compression_type = ""
    name = None
    input_dict = {"bytes": bytes_data, "compression_type": compression_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Simple ZLIB compressed string
    data = b"Hello, world!"
    compressed_data = zlib.compress(data)
    bytes_data = np.array([compressed_data], dtype=np.object_)
    compression_type = "ZLIB"
    name = "zlib_test"
    input_dict = {"bytes": bytes_data, "compression_type": compression_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Simple GZIP compressed string
    data = b"This is a test string for GZIP compression."
    compressed_data = gzip.compress(data)
    bytes_data = np.array([compressed_data], dtype=np.object_)
    compression_type = "GZIP"
    name = "gzip_test"
    input_dict = {"bytes": bytes_data, "compression_type": compression_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multiple ZLIB compressed strings
    data1 = b"String 1"
    data2 = b"String 2"
    compressed_data1 = zlib.compress(data1)
    compressed_data2 = zlib.compress(data2)
    bytes_data = np.array([compressed_data1, compressed_data2], dtype=np.object_)
    compression_type = "ZLIB"
    name = None
    input_dict = {"bytes": bytes_data, "compression_type": compression_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multiple GZIP compressed strings
    data1 = b"String A"
    data2 = b"String B"
    compressed_data1 = gzip.compress(data1)
    compressed_data2 = gzip.compress(data2)
    bytes_data = np.array([compressed_data1, compressed_data2], dtype=np.object_)
    compression_type = "GZIP"
    name = None
    input_dict = {"bytes": bytes_data, "compression_type": compression_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: ZLIB compressed, with unicode
    data = "你好，世界！".encode('utf-8')
    compressed_data = zlib.compress(data)
    bytes_data = np.array([compressed_data], dtype=np.object_)
    compression_type = "ZLIB"
    name = None
    input_dict = {"bytes": bytes_data, "compression_type": compression_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: GZIP compressed, with unicode
    data = "こんにちは世界".encode('utf-8')
    compressed_data = gzip.compress(data)
    bytes_data = np.array([compressed_data], dtype=np.object_)
    compression_type = "GZIP"
    name = None
    input_dict = {"bytes": bytes_data, "compression_type": compression_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty numpy array
    bytes_data = np.array([], dtype=np.object_)
    compression_type = ""
    name = None
    input_dict = {"bytes": bytes_data, "compression_type": compression_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: ZLIB compressed with a longer string
    data = b"This is a much longer string to test ZLIB compression with more data. Let's see if it works as expected."
    compressed_data = zlib.compress(data)
    bytes_data = np.array([compressed_data], dtype=np.object_)
    compression_type = "ZLIB"
    name = None
    input_dict = {"bytes": bytes_data, "compression_type": compression_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: GZIP compressed with a longer string
    data = b"This is a much longer string to test GZIP compression with more data. It should be able to handle this without any issues."
    compressed_data = gzip.compress(data)
    bytes_data = np.array([compressed_data], dtype=np.object_)
    compression_type = "GZIP"
    name = None
    input_dict = {"bytes": bytes_data, "compression_type": compression_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Multiple Empty string with ZLIB
    bytes_data = np.array([b"", b""], dtype=np.object_)
    compression_type = "ZLIB"
    name = None
    input_dict = {"bytes": bytes_data, "compression_type": compression_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Multiple Empty string with GZIP
    bytes_data = np.array([b"", b""], dtype=np.object_)
    compression_type = "GZIP"
    name = None
    input_dict = {"bytes": bytes_data, "compression_type": compression_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.DecodeCompressed"] = tf_raw_ops_decodecompressed_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DecodeCompressed' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DecodeCompressed'.")

check_valid('tf.raw_ops.DecodeCompressed', generated_inputs['tf.raw_ops.DecodeCompressed'], lib="tf", suffix=0)
