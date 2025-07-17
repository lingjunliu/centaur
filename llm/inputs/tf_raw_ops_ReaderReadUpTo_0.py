
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ReaderReadUpTo_inputs():
    list_of_inputs = []

    # Input 1
    reader_handle = np.array("reader_handle").astype(np.object_)
    queue_handle = np.array("queue_handle").astype(np.object_)
    num_records = np.array(5).astype(np.int64)
    input_dict = {"reader_handle": reader_handle, "queue_handle": queue_handle, "num_records": num_records, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    reader_handle = np.array("reader_handle2").astype(np.object_)
    queue_handle = np.array("queue_handle2").astype(np.object_)
    num_records = np.array(10).astype(np.int64)
    input_dict = {"reader_handle": reader_handle, "queue_handle": queue_handle, "num_records": num_records, "name": "op_name"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    reader_handle = np.array("reader_handle3").astype(np.object_)
    queue_handle = np.array("queue_handle3").astype(np.object_)
    num_records = np.array(1).astype(np.int64)
    input_dict = {"reader_handle": reader_handle, "queue_handle": queue_handle, "num_records": num_records, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    reader_handle = np.array("reader_handle4").astype(np.object_)
    queue_handle = np.array("queue_handle4").astype(np.object_)
    num_records = np.array(0).astype(np.int64)
    input_dict = {"reader_handle": reader_handle, "queue_handle": queue_handle, "num_records": num_records, "name": "op_name4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    reader_handle = np.array("reader_handle5").astype(np.object_)
    queue_handle = np.array("queue_handle5").astype(np.object_)
    num_records = np.array(100).astype(np.int64)
    input_dict = {"reader_handle": reader_handle, "queue_handle": queue_handle, "num_records": num_records, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    reader_handle = np.array("reader_handle6").astype(np.object_)
    queue_handle = np.array("queue_handle6").astype(np.object_)
    num_records = np.array(1024).astype(np.int64)
    input_dict = {"reader_handle": reader_handle, "queue_handle": queue_handle, "num_records": num_records, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    reader_handle = np.array("reader_handle7").astype(np.object_)
    queue_handle = np.array("queue_handle7").astype(np.object_)
    num_records = np.array(2**20).astype(np.int64)
    input_dict = {"reader_handle": reader_handle, "queue_handle": queue_handle, "num_records": num_records, "name": "op_name7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    reader_handle = np.array("reader_handle8").astype(np.object_)
    queue_handle = np.array("queue_handle8").astype(np.object_)
    num_records = np.array(2).astype(np.int64)
    input_dict = {"reader_handle": reader_handle, "queue_handle": queue_handle, "num_records": num_records, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    reader_handle = np.array("reader_handle9").astype(np.object_)
    queue_handle = np.array("queue_handle9").astype(np.object_)
    num_records = np.array(64).astype(np.int64)
    input_dict = {"reader_handle": reader_handle, "queue_handle": queue_handle, "num_records": num_records, "name": "op_name9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    reader_handle = np.array("reader_handle10").astype(np.object_)
    queue_handle = np.array("queue_handle10").astype(np.object_)
    num_records = np.array(2048).astype(np.int64)
    input_dict = {"reader_handle": reader_handle, "queue_handle": queue_handle, "num_records": num_records, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ReaderReadUpTo"] = tf_raw_ops_ReaderReadUpTo_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ReaderReadUpTo' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ReaderReadUpTo'.")

check_valid('tf.raw_ops.ReaderReadUpTo', generated_inputs['tf.raw_ops.ReaderReadUpTo'], lib="tf", suffix=0)
