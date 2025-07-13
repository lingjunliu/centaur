
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_reader_read_up_to_inputs():
    list_of_inputs = []

    # Input 1
    reader_handle = tf.compat.v1.train.string_input_producer(["test"]).queue.dequeue()
    queue_handle = tf.compat.v1.train.string_input_producer(["test"]).queue.dequeue()
    num_records = tf.constant(10, dtype=tf.int64)
    name = "op_1"
    input_dict = {"reader_handle": reader_handle, "queue_handle": queue_handle, "num_records": num_records, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    reader_handle = tf.compat.v1.train.string_input_producer(["test"]).queue.dequeue()
    queue_handle = tf.compat.v1.train.string_input_producer(["test"]).queue.dequeue()
    num_records = tf.constant(1, dtype=tf.int64)
    name = "op_2"
    input_dict = {"reader_handle": reader_handle, "queue_handle": queue_handle, "num_records": num_records, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    reader_handle = tf.compat.v1.train.string_input_producer(["test"]).queue.dequeue()
    queue_handle = tf.compat.v1.train.string_input_producer(["test"]).queue.dequeue()
    num_records = tf.constant(5, dtype=tf.int64)
    name = None
    input_dict = {"reader_handle": reader_handle, "queue_handle": queue_handle, "num_records": num_records, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    reader_handle = tf.compat.v1.train.string_input_producer(["test"]).queue.dequeue()
    queue_handle = tf.compat.v1.train.string_input_producer(["test"]).queue.dequeue()
    num_records = tf.constant(0, dtype=tf.int64)
    name = "op_4"
    input_dict = {"reader_handle": reader_handle, "queue_handle": queue_handle, "num_records": num_records, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    reader_handle = tf.compat.v1.train.string_input_producer(["test"]).queue.dequeue()
    queue_handle = tf.compat.v1.train.string_input_producer(["test"]).queue.dequeue()
    num_records = tf.constant(100, dtype=tf.int64)
    name = None
    input_dict = {"reader_handle": reader_handle, "queue_handle": queue_handle, "num_records": num_records, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    reader_handle = tf.compat.v1.train.string_input_producer(["test"]).queue.dequeue()
    queue_handle = tf.compat.v1.train.string_input_producer(["test"]).queue.dequeue()
    num_records = tf.constant(20, dtype=tf.int64)
    name = "op_6"
    input_dict = {"reader_handle": reader_handle, "queue_handle": queue_handle, "num_records": num_records, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7
    reader_handle = tf.compat.v1.train.string_input_producer(["test"]).queue.dequeue()
    queue_handle = tf.compat.v1.train.string_input_producer(["test"]).queue.dequeue()
    num_records = tf.constant(2, dtype=tf.int64)
    name = None
    input_dict = {"reader_handle": reader_handle, "queue_handle": queue_handle, "num_records": num_records, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    reader_handle = tf.compat.v1.train.string_input_producer(["test"]).queue.dequeue()
    queue_handle = tf.compat.v1.train.string_input_producer(["test"]).queue.dequeue()
    num_records = tf.constant(7, dtype=tf.int64)
    name = "op_8"
    input_dict = {"reader_handle": reader_handle, "queue_handle": queue_handle, "num_records": num_records, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    reader_handle = tf.compat.v1.train.string_input_producer(["test"]).queue.dequeue()
    queue_handle = tf.compat.v1.train.string_input_producer(["test"]).queue.dequeue()
    num_records = tf.constant(15, dtype=tf.int64)
    name = None
    input_dict = {"reader_handle": reader_handle, "queue_handle": queue_handle, "num_records": num_records, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    reader_handle = tf.compat.v1.train.string_input_producer(["test"]).queue.dequeue()
    queue_handle = tf.compat.v1.train.string_input_producer(["test"]).queue.dequeue()
    num_records = tf.constant(30, dtype=tf.int64)
    name = "op_10"
    input_dict = {"reader_handle": reader_handle, "queue_handle": queue_handle, "num_records": num_records, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ReaderReadUpTo"] = tf_raw_ops_reader_read_up_to_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ReaderReadUpTo' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ReaderReadUpTo'.")

check_valid('tf.raw_ops.ReaderReadUpTo', generated_inputs['tf.raw_ops.ReaderReadUpTo'], lib="tf", suffix=0)
