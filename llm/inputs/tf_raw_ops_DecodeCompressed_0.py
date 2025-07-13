
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
import zlib
import gzip
import io

def tf_raw_ops_decode_compressed_inputs():
    list_of_inputs = []

    # Input 1: Empty string, no compression
    bytes_data = np.array([b""], dtype=np.object_)
    input_dict = {"bytes": bytes_data, "compression_type": "", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Simple string, no compression
    bytes_data = np.array([b"hello"], dtype=np.object_)
    input_dict = {"bytes": bytes_data, "compression_type": "", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: ZLIB compressed string
    data = b"This is a test string to be compressed with ZLIB."
    compressed_data = zlib.compress(data)
    bytes_data = np.array([compressed_data], dtype=np.object_)
    input_dict = {"bytes": bytes_data, "compression_type": "ZLIB", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: GZIP compressed string
    data = b"This is a test string to be compressed with GZIP."
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb') as f:
        f.write(data)
    compressed_data = buf.getvalue()
    bytes_data = np.array([compressed_data], dtype=np.object_)
    input_dict = {"bytes": bytes_data, "compression_type": "GZIP", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multiple strings, no compression
    bytes_data = np.array([b"hello", b"world", b"!"], dtype=np.object_)
    input_dict = {"bytes": bytes_data, "compression_type": "", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multiple ZLIB compressed strings
    data1 = b"String 1 compressed with ZLIB."
    compressed_data1 = zlib.compress(data1)
    data2 = b"String 2 compressed with ZLIB."
    compressed_data2 = zlib.compress(data2)
    bytes_data = np.array([compressed_data1, compressed_data2], dtype=np.object_)
    input_dict = {"bytes": bytes_data, "compression_type": "ZLIB", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Multiple GZIP compressed strings
    data1 = b"String 1 compressed with GZIP."
    buf1 = io.BytesIO()
    with gzip.GzipFile(fileobj=buf1, mode='wb') as f:
        f.write(data1)
    compressed_data1 = buf1.getvalue()
    data2 = b"String 2 compressed with GZIP."
    buf2 = io.BytesIO()
    with gzip.GzipFile(fileobj=buf2, mode='wb') as f:
        f.write(data2)
    compressed_data2 = buf2.getvalue()
    bytes_data = np.array([compressed_data1, compressed_data2], dtype=np.object_)
    input_dict = {"bytes": bytes_data, "compression_type": "GZIP", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D array of strings, no compression
    bytes_data = np.array([[b"hello", b"world"], [b"!", b"test"]], dtype=np.object_)
    input_dict = {"bytes": bytes_data, "compression_type": "", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: ZLIB compressed string in 2D array
    data1 = b"String 1 compressed with ZLIB."
    compressed_data1 = zlib.compress(data1)
    data2 = b"String 2 compressed with ZLIB."
    compressed_data2 = zlib.compress(data2)
    bytes_data = np.array([[compressed_data1, compressed_data2], [compressed_data1, compressed_data2]], dtype=np.object_)

    input_dict = {"bytes": bytes_data, "compression_type": "ZLIB", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: GZIP compressed string in 2D array
    data1 = b"String 1 compressed with GZIP."
    buf1 = io.BytesIO()
    with gzip.GzipFile(fileobj=buf1, mode='wb') as f:
        f.write(data1)
    compressed_data1 = buf1.getvalue()
    data2 = b"String 2 compressed with GZIP."
    buf2 = io.BytesIO()
    with gzip.GzipFile(fileobj=buf2, mode='wb') as f:
        f.write(data2)
    compressed_data2 = buf2.getvalue()
    bytes_data = np.array([[compressed_data1, compressed_data2], [compressed_data1, compressed_data2]], dtype=np.object_)
    input_dict = {"bytes": bytes_data, "compression_type": "GZIP", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DecodeCompressed' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DecodeCompressed'.")

check_valid('tf.raw_ops.DecodeCompressed', generated_inputs['tf.raw_ops.DecodeCompressed'], lib="tf", suffix=0)
