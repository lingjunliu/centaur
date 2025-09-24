
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ReaderSerializeState_inputs():
    list_of_inputs = []

    # Input 1
    try:
      reader = tf.compat.v1.TFRecordReader()
      input_dict = {"reader_handle": reader.reader_handle, "name": None}
      list_of_inputs.append(copy.deepcopy(input_dict))
    except:
      pass

    # Input 2
    try:
      reader = tf.compat.v1.TFRecordReader()
      input_dict = {"reader_handle": reader.reader_handle, "name": "serialize_state"}
      list_of_inputs.append(copy.deepcopy(input_dict))
    except:
      pass

    # Input 3
    try:
        reader = tf.compat.v1.TFRecordReader()
        input_dict = {"reader_handle": reader.reader_handle, "name": None}
        list_of_inputs.append(copy.deepcopy(input_dict))
    except:
        pass

    # Input 4
    try:
        reader = tf.compat.v1.TFRecordReader()
        input_dict = {"reader_handle": reader.reader_handle, "name": "another_name"}
        list_of_inputs.append(copy.deepcopy(input_dict))
    except:
        pass
    
    # Input 5
    try:
        reader = tf.compat.v1.TFRecordReader()
        input_dict = {"reader_handle": reader.reader_handle, "name": ""}
        list_of_inputs.append(copy.deepcopy(input_dict))
    except:
        pass

    # Input 6
    try:
      reader = tf.compat.v1.TFRecordReader()
      input_dict = {"reader_handle": reader.reader_handle, "name": "serialize_state_2"}
      list_of_inputs.append(copy.deepcopy(input_dict))
    except:
      pass

    # Input 7
    try:
      reader = tf.compat.v1.TFRecordReader()
      input_dict = {"reader_handle": reader.reader_handle, "name": "name123"}
      list_of_inputs.append(copy.deepcopy(input_dict))
    except:
      pass
    
    # Input 8
    try:
      reader = tf.compat.v1.TFRecordReader()
      input_dict = {"reader_handle": reader.reader_handle, "name": "reader_state"}
      list_of_inputs.append(copy.deepcopy(input_dict))
    except:
      pass

    # Input 9
    try:
      reader = tf.compat.v1.TFRecordReader()
      input_dict = {"reader_handle": reader.reader_handle, "name": "state_reader"}
      list_of_inputs.append(copy.deepcopy(input_dict))
    except:
      pass
    
    # Input 10
    try:
      reader = tf.compat.v1.TFRecordReader()
      input_dict = {"reader_handle": reader.reader_handle, "name": "data_reader"}
      list_of_inputs.append(copy.deepcopy(input_dict))
    except:
      pass

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ReaderSerializeState"] = tf_raw_ops_ReaderSerializeState_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ReaderSerializeState' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ReaderSerializeState'.")

check_valid('tf.raw_ops.ReaderSerializeState', generated_inputs['tf.raw_ops.ReaderSerializeState'], lib="tf", suffix=0)
