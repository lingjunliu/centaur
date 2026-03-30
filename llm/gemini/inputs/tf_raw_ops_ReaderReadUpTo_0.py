
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ReaderReadUpTo_inputs():
    list_of_inputs = []

    # Input 1
    reader_handle = tf.compat.v1.placeholder(dtype=tf.string)
    queue_handle = tf.compat.v1.placeholder(dtype=tf.string)
    num_records = tf.constant(1, dtype=tf.int64)
    input_dict = {"reader_handle": reader_handle, "queue_handle": queue_handle, "num_records": num_records, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    reader_handle = tf.compat.v1.placeholder(dtype=tf.string)
    queue_handle = tf.compat.v1.placeholder(dtype=tf.string)
    num_records = tf.constant(5, dtype=tf.int64)
    input_dict = {"reader_handle": reader_handle, "queue_handle": queue_handle, "num_records": num_records, "name": "test_reader_read_up_to"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    reader_handle = tf.compat.v1.placeholder(dtype=tf.string)
    queue_handle = tf.compat.v1.placeholder(dtype=tf.string)
    num_records = tf.constant(10, dtype=tf.int64)
    input_dict = {"reader_handle": reader_handle, "queue_handle": queue_handle, "num_records": num_records, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    reader_handle = tf.compat.v1.placeholder(dtype=tf.string)
    queue_handle = tf.compat.v1.placeholder(dtype=tf.string)
    num_records = tf.constant(20, dtype=tf.int64)
    input_dict = {"reader_handle": reader_handle, "queue_handle": queue_handle, "num_records": num_records, "name": "my_reader"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    reader_handle = tf.compat.v1.placeholder(dtype=tf.string)
    queue_handle = tf.compat.v1.placeholder(dtype=tf.string)
    num_records = tf.constant(100, dtype=tf.int64)
    input_dict = {"reader_handle": reader_handle, "queue_handle": queue_handle, "num_records": num_records, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6
    reader_handle = tf.compat.v1.placeholder(dtype=tf.string)
    queue_handle = tf.compat.v1.placeholder(dtype=tf.string)
    num_records = tf.constant(2, dtype=tf.int64)
    input_dict = {"reader_handle": reader_handle, "queue_handle": queue_handle, "num_records": num_records, "name": "reader_test"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    reader_handle = tf.compat.v1.placeholder(dtype=tf.string)
    queue_handle = tf.compat.v1.placeholder(dtype=tf.string)
    num_records = tf.constant(3, dtype=tf.int64)
    input_dict = {"reader_handle": reader_handle, "queue_handle": queue_handle, "num_records": num_records, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    reader_handle = tf.compat.v1.placeholder(dtype=tf.string)
    queue_handle = tf.compat.v1.placeholder(dtype=tf.string)
    num_records = tf.constant(4, dtype=tf.int64)
    input_dict = {"reader_handle": reader_handle, "queue_handle": queue_handle, "num_records": num_records, "name": "test_name"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    reader_handle = tf.compat.v1.placeholder(dtype=tf.string)
    queue_handle = tf.compat.v1.placeholder(dtype=tf.string)
    num_records = tf.constant(7, dtype=tf.int64)
    input_dict = {"reader_handle": reader_handle, "queue_handle": queue_handle, "num_records": num_records, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    reader_handle = tf.compat.v1.placeholder(dtype=tf.string)
    queue_handle = tf.compat.v1.placeholder(dtype=tf.string)
    num_records = tf.constant(12, dtype=tf.int64)
    input_dict = {"reader_handle": reader_handle, "queue_handle": queue_handle, "num_records": num_records, "name": "example_name"}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ReaderReadUpTo"] = tf_raw_ops_ReaderReadUpTo_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ReaderReadUpTo' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ReaderReadUpTo'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.ReaderReadUpTo', generated_inputs['tf.raw_ops.ReaderReadUpTo'], lib="tf", suffix=0)
