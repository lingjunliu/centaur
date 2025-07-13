
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ReaderSerializeState_inputs():
    list_of_inputs = []

    # Input 1: Simple string reader handle
    reader_handle = tf.constant("my_reader_handle")
    input_dict = {"reader_handle": reader_handle, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Reader handle with a name
    reader_handle = tf.constant("another_reader")
    input_dict = {"reader_handle": reader_handle, "name": "my_reader_state"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Another reader handle, empty string
    reader_handle = tf.constant("")
    input_dict = {"reader_handle": reader_handle, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Reader handle with a longer name
    reader_handle = tf.constant("yet_another_reader")
    input_dict = {"reader_handle": reader_handle, "name": "a_very_long_and_descriptive_name"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Reader handle created using a numpy array
    reader_handle = tf.constant(np.array("my_handle").astype(np.string_))
    input_dict = {"reader_handle": reader_handle, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different reader handle string
    reader_handle = tf.constant("different_handle")
    input_dict = {"reader_handle": reader_handle, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Reader handle with special chars in name
    reader_handle = tf.constant("special_reader")
    input_dict = {"reader_handle": reader_handle, "name": "reader.state-123"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Reader handle initialized to a dummy value using constant
    reader_handle = tf.constant("dummy_value")
    input_dict = {"reader_handle": reader_handle, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Name as an empty string
    reader_handle = tf.constant("empty_name_reader")
    input_dict = {"reader_handle": reader_handle, "name": ""}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Reader handle from numpy and slightly different name
    reader_handle = tf.constant(np.array("stateful_reader").astype(np.string_))
    input_dict = {"reader_handle": reader_handle, "name": "serialised_state"}
    list_of_inputs.append(copy.deepcopy(input_dict))

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
