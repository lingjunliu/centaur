
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_data_experimental_tfrecordwriter_inputs():
    list_of_inputs = []

    # Input 1: Basic filename, no compression
    input_dict = {
        "filename": "/tmp/test1.tfrecord",
        "compression_type": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Filename with .gz extension, gzip compression
    input_dict = {
        "filename": "/tmp/test2.tfrecord.gz",
        "compression_type": "GZIP"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Filename with .zlib extension, zlib compression
    input_dict = {
        "filename": "/tmp/test3.tfrecord.zlib",
        "compression_type": "ZLIB"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different filename, GZIP
    input_dict = {
        "filename": "/tmp/data/another_file.tfrecord.gz",
        "compression_type": "GZIP"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Another filename, ZLIB
    input_dict = {
        "filename": "/tmp/data/zlib_file.tfrecord.zlib",
        "compression_type": "ZLIB"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Relative path filename, no compression
    input_dict = {
        "filename": "relative_path.tfrecord",
        "compression_type": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty filename, no compression (may raise an error during write but it's a valid input)
    input_dict = {
        "filename": "",
        "compression_type": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Unicode filename, gzip compression
    input_dict = {
        "filename": "/tmp/你好世界.tfrecord.gz",
        "compression_type": "GZIP"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: filename with spaces, no compression
    input_dict = {
        "filename": "/tmp/my file.tfrecord",
        "compression_type": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Filename with special chars
    input_dict = {
        "filename": "/tmp/file!@#$.tfrecord",
        "compression_type": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.TFRecordWriter"] = tf_data_experimental_tfrecordwriter_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.data.experimental.TFRecordWriter' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.TFRecordWriter'.")

check_valid('tf.data.experimental.TFRecordWriter', generated_inputs['tf.data.experimental.TFRecordWriter'], lib="tf", suffix=0)
