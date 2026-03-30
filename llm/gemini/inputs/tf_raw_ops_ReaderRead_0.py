
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ReaderRead_inputs():
    list_of_inputs = []

    # Input 1
    reader_handle = np.array("reader_handle_1", dtype=np.string_)
    queue_handle = np.array("queue_handle_1", dtype=np.string_)
    name = "reader_read_1"

    input_dict = {
        "reader_handle": reader_handle,
        "queue_handle": queue_handle,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    reader_handle = np.array("reader_handle_2", dtype=np.string_)
    queue_handle = np.array("queue_handle_2", dtype=np.string_)
    name = None

    input_dict = {
        "reader_handle": reader_handle,
        "queue_handle": queue_handle,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    reader_handle = np.array("reader_handle_3", dtype=np.string_)
    queue_handle = np.array("queue_handle_3", dtype=np.string_)
    name = "reader_read_3"

    input_dict = {
        "reader_handle": reader_handle,
        "queue_handle": queue_handle,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    reader_handle = np.array("reader_handle_4", dtype=np.string_)
    queue_handle = np.array("queue_handle_4", dtype=np.string_)
    name = None

    input_dict = {
        "reader_handle": reader_handle,
        "queue_handle": queue_handle,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    reader_handle = np.array("reader_handle_5", dtype=np.string_)
    queue_handle = np.array("queue_handle_5", dtype=np.string_)
    name = "reader_read_5"

    input_dict = {
        "reader_handle": reader_handle,
        "queue_handle": queue_handle,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    reader_handle = np.array("reader_handle_6", dtype=np.string_)
    queue_handle = np.array("queue_handle_6", dtype=np.string_)
    name = None

    input_dict = {
        "reader_handle": reader_handle,
        "queue_handle": queue_handle,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    reader_handle = np.array("reader_handle_7", dtype=np.string_)
    queue_handle = np.array("queue_handle_7", dtype=np.string_)
    name = "reader_read_7"

    input_dict = {
        "reader_handle": reader_handle,
        "queue_handle": queue_handle,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    reader_handle = np.array("reader_handle_8", dtype=np.string_)
    queue_handle = np.array("queue_handle_8", dtype=np.string_)
    name = None

    input_dict = {
        "reader_handle": reader_handle,
        "queue_handle": queue_handle,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    reader_handle = np.array("reader_handle_9", dtype=np.string_)
    queue_handle = np.array("queue_handle_9", dtype=np.string_)
    name = "reader_read_9"

    input_dict = {
        "reader_handle": reader_handle,
        "queue_handle": queue_handle,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    reader_handle = np.array("reader_handle_10", dtype=np.string_)
    queue_handle = np.array("queue_handle_10", dtype=np.string_)
    name = None

    input_dict = {
        "reader_handle": reader_handle,
        "queue_handle": queue_handle,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}

generated_inputs["tf.raw_ops.ReaderRead"] = tf_raw_ops_ReaderRead_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ReaderRead' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ReaderRead'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.ReaderRead', generated_inputs['tf.raw_ops.ReaderRead'], lib="tf", suffix=0)
