
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_reader_num_records_produced_inputs():
    list_of_inputs = []

    # Create a dummy tfrecord file
    filename = "dummy.tfrecord"
    writer = tf.io.TFRecordWriter(filename)
    example = tf.train.Example()
    writer.write(example.SerializeToString())
    writer.close()


    # Input 1
    try:
        reader_handle = tf.compat.v1.io.tf_record_iterator(filename).reader.reader_handle()
    except Exception as e:
        print(f"Error creating reader_handle: {e}")
        return []

    name = None
    input_dict = {"reader_handle": reader_handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    try:
        reader_handle = tf.compat.v1.io.tf_record_iterator(filename).reader.reader_handle()
    except:
        return []
    name = "my_reader_op"
    input_dict = {"reader_handle": reader_handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    try:
        reader_handle = tf.compat.v1.io.tf_record_iterator(filename).reader.reader_handle()
    except:
        return []
    name = ""
    input_dict = {"reader_handle": reader_handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    try:
        reader_handle = tf.compat.v1.io.tf_record_iterator(filename).reader.reader_handle()
    except:
        return []
    name = "AnotherReaderOp"
    input_dict = {"reader_handle": reader_handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    try:
        reader_handle = tf.compat.v1.io.tf_record_iterator(filename).reader.reader_handle()
    except:
        return []
    name = "Op_5"
    input_dict = {"reader_handle": reader_handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    try:
        reader_handle = tf.compat.v1.io.tf_record_iterator(filename).reader.reader_handle()
    except:
        return []
    name = "VeryLongNameForAnOp_1234567890"
    input_dict = {"reader_handle": reader_handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    try:
        reader_handle = tf.compat.v1.io.tf_record_iterator(filename).reader.reader_handle()
    except:
        return []
    name = "ReaderWithNumbers123"
    input_dict = {"reader_handle": reader_handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    try:
        reader_handle = tf.compat.v1.io.tf_record_iterator(filename).reader.reader_handle()
    except:
        return []
    name = "SpecialCharsReader!@#$"
    input_dict = {"reader_handle": reader_handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    try:
        reader_handle = tf.compat.v1.io.tf_record_iterator(filename).reader.reader_handle()
    except:
        return []
    name = "ReaderUpperCase"
    input_dict = {"reader_handle": reader_handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 10
    try:
        reader_handle = tf.compat.v1.io.tf_record_iterator(filename).reader.reader_handle()
    except:
        return []
    name = "ReaderLowerCase"
    input_dict = {"reader_handle": reader_handle, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ReaderNumRecordsProduced"] = tf_raw_ops_reader_num_records_produced_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ReaderNumRecordsProduced' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ReaderNumRecordsProduced'.")

check_valid('tf.raw_ops.ReaderNumRecordsProduced', generated_inputs['tf.raw_ops.ReaderNumRecordsProduced'], lib="tf", suffix=0)
