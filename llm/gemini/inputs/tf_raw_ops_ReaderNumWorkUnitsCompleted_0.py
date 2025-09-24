
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ReaderNumWorkUnitsCompleted_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    try:
        reader_handle = tf.compat.v1.io.tf_record_iterator("dummy.tfrecord").reader_handle()
        input_dict = {"reader_handle": reader_handle, "name": None}
        list_of_inputs.append(copy.deepcopy(input_dict))
    except:
        pass

    # Input 2: With a name
    try:
        reader_handle = tf.compat.v1.io.tf_record_iterator("dummy.tfrecord").reader_handle()
        input_dict = {"reader_handle": reader_handle, "name": "my_reader"}
        list_of_inputs.append(copy.deepcopy(input_dict))
    except:
        pass

    # Input 3: Empty string for name
    try:
        reader_handle = tf.compat.v1.io.tf_record_iterator("dummy.tfrecord").reader_handle()
        input_dict = {"reader_handle": reader_handle, "name": ""}
        list_of_inputs.append(copy.deepcopy(input_dict))
    except:
        pass

    # Input 4: Name with special characters
    try:
        reader_handle = tf.compat.v1.io.tf_record_iterator("dummy.tfrecord").reader_handle()
        input_dict = {"reader_handle": reader_handle, "name": "my_reader!@#"}
        list_of_inputs.append(copy.deepcopy(input_dict))
    except:
        pass

    # Input 5: Long name
    try:
        reader_handle = tf.compat.v1.io.tf_record_iterator("dummy.tfrecord").reader_handle()
        long_name = "a" * 200
        input_dict = {"reader_handle": reader_handle, "name": long_name}
        list_of_inputs.append(copy.deepcopy(input_dict))
    except:
        pass

    # Input 6: Reader handle to a different file
    try:
        reader_handle = tf.compat.v1.io.tf_record_iterator("dummy.tfrecord").reader_handle()
        input_dict = {"reader_handle": reader_handle, "name": None}
        list_of_inputs.append(copy.deepcopy(input_dict))
    except:
        pass

    # Input 7: Another valid name
    try:
        reader_handle = tf.compat.v1.io.tf_record_iterator("dummy.tfrecord").reader_handle()
        input_dict = {"reader_handle": reader_handle, "name": "reader_name_2"}
        list_of_inputs.append(copy.deepcopy(input_dict))
    except:
        pass

    # Input 8: Unicode name
    try:
        reader_handle = tf.compat.v1.io.tf_record_iterator("dummy.tfrecord").reader_handle()
        input_dict = {"reader_handle": reader_handle, "name": "你好"}
        list_of_inputs.append(copy.deepcopy(input_dict))
    except:
        pass

    # Input 9: Name with numbers
    try:
        reader_handle = tf.compat.v1.io.tf_record_iterator("dummy.tfrecord").reader_handle()
        input_dict = {"reader_handle": reader_handle, "name": "reader123"}
        list_of_inputs.append(copy.deepcopy(input_dict))
    except:
        pass

    # Input 10: Name with mixed case
    try:
        reader_handle = tf.compat.v1.io.tf_record_iterator("dummy.tfrecord").reader_handle()
        input_dict = {"reader_handle": reader_handle, "name": "MyReaderName"}
        list_of_inputs.append(copy.deepcopy(input_dict))
    except:
        pass

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ReaderNumWorkUnitsCompleted"] = tf_raw_ops_ReaderNumWorkUnitsCompleted_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ReaderNumWorkUnitsCompleted' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ReaderNumWorkUnitsCompleted'.")

check_valid('tf.raw_ops.ReaderNumWorkUnitsCompleted', generated_inputs['tf.raw_ops.ReaderNumWorkUnitsCompleted'], lib="tf", suffix=0)
