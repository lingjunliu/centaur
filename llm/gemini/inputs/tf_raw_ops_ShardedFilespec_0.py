
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_sharded_filespec_inputs():
    list_of_inputs = []

    # Input 1
    basename = np.array("data", dtype=np.string_)
    num_shards = np.array(5, dtype=np.int32)
    name = "sharded_filespec_1"
    input_dict = {"basename": basename, "num_shards": num_shards, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    basename = np.array("train", dtype=np.string_)
    num_shards = np.array(10, dtype=np.int32)
    name = "sharded_filespec_2"
    input_dict = {"basename": basename, "num_shards": num_shards, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    basename = np.array("eval", dtype=np.string_)
    num_shards = np.array(1, dtype=np.int32)
    name = "sharded_filespec_3"
    input_dict = {"basename": basename, "num_shards": num_shards, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    basename = np.array("model", dtype=np.string_)
    num_shards = np.array(20, dtype=np.int32)
    name = "sharded_filespec_4"
    input_dict = {"basename": basename, "num_shards": num_shards, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    basename = np.array("test", dtype=np.string_)
    num_shards = np.array(100, dtype=np.int32)
    name = "sharded_filespec_5"
    input_dict = {"basename": basename, "num_shards": num_shards, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    basename = np.array("images", dtype=np.string_)
    num_shards = np.array(2, dtype=np.int32)
    name = "sharded_filespec_6"
    input_dict = {"basename": basename, "num_shards": num_shards, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    basename = np.array("logs", dtype=np.string_)
    num_shards = np.array(3, dtype=np.int32)
    name = "sharded_filespec_7"
    input_dict = {"basename": basename, "num_shards": num_shards, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    basename = np.array("checkpoints", dtype=np.string_)
    num_shards = np.array(7, dtype=np.int32)
    name = "sharded_filespec_8"
    input_dict = {"basename": basename, "num_shards": num_shards, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    basename = np.array("output", dtype=np.string_)
    num_shards = np.array(12, dtype=np.int32)
    name = "sharded_filespec_9"
    input_dict = {"basename": basename, "num_shards": num_shards, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    basename = np.array("results", dtype=np.string_)
    num_shards = np.array(32, dtype=np.int32)
    name = "sharded_filespec_10"
    input_dict = {"basename": basename, "num_shards": num_shards, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ShardedFilespec"] = tf_raw_ops_sharded_filespec_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ShardedFilespec' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ShardedFilespec'.")

check_valid('tf.raw_ops.ShardedFilespec', generated_inputs['tf.raw_ops.ShardedFilespec'], lib="tf", suffix=0)
