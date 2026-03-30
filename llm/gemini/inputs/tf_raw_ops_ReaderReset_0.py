
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_reader_reset_inputs():
    list_of_inputs = []

    # Input 1
    reader = tf.compat.v1.io.TFRecordReader()
    reader_handle = reader.reader_handle
    input_dict = {"reader_handle": reader_handle, "name": "reset_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    reader = tf.compat.v1.io.TFRecordReader()
    reader_handle = reader.reader_handle
    input_dict = {"reader_handle": reader_handle, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    reader = tf.compat.v1.io.TFRecordReader()
    reader_handle = reader.reader_handle
    input_dict = {"reader_handle": reader_handle, "name": "reset_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    reader = tf.compat.v1.io.TextLineReader()
    reader_handle = reader.reader_handle
    input_dict = {"reader_handle": reader_handle, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    reader = tf.compat.v1.io.FixedLengthRecordReader(record_bytes=10)
    reader_handle = reader.reader_handle
    input_dict = {"reader_handle": reader_handle, "name": "reset_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    reader = tf.compat.v1.io.WholeFileReader()
    reader_handle = reader.reader_handle
    input_dict = {"reader_handle": reader_handle, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    reader = tf.compat.v1.io.IdentityReader()
    reader_handle = reader.reader_handle
    input_dict = {"reader_handle": reader_handle, "name": "reset_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    reader = tf.compat.v1.io.TextLineReader()
    reader_handle = reader.reader_handle
    input_dict = {"reader_handle": reader_handle, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    reader = tf.compat.v1.io.FixedLengthRecordReader(record_bytes=20)
    reader_handle = reader.reader_handle
    input_dict = {"reader_handle": reader_handle, "name": "reset_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    reader = tf.compat.v1.io.WholeFileReader()
    reader_handle = reader.reader_handle
    input_dict = {"reader_handle": reader_handle, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ReaderReset"] = tf_raw_ops_reader_reset_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ReaderReset' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ReaderReset'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.ReaderReset', generated_inputs['tf.raw_ops.ReaderReset'], lib="tf", suffix=0)
