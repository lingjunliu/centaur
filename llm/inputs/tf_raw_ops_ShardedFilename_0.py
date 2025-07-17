
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ShardedFilename_inputs():
    list_of_inputs = []

    # Input 1
    basename = np.array("file", dtype=np.dtype('string'))
    shard = np.array(0, dtype=np.int32)
    num_shards = np.array(1, dtype=np.int32)
    name = "shard_file_1"
    input_dict = {"basename": basename, "shard": shard, "num_shards": num_shards, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    basename = np.array("data", dtype=np.dtype('string'))
    shard = np.array(5, dtype=np.int32)
    num_shards = np.array(10, dtype=np.int32)
    name = "shard_file_2"
    input_dict = {"basename": basename, "shard": shard, "num_shards": num_shards, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    basename = np.array("output", dtype=np.dtype('string'))
    shard = np.array(99, dtype=np.int32)
    num_shards = np.array(100, dtype=np.int32)
    name = "shard_file_3"
    input_dict = {"basename": basename, "shard": shard, "num_shards": num_shards, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    basename = np.array("my_model", dtype=np.dtype('string'))
    shard = np.array(1234, dtype=np.int32)
    num_shards = np.array(5678, dtype=np.int32)
    name = "shard_file_4"
    input_dict = {"basename": basename, "shard": shard, "num_shards": num_shards, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    basename = np.array("training_data", dtype=np.dtype('string'))
    shard = np.array(1, dtype=np.int32)
    num_shards = np.array(1000, dtype=np.int32)
    name = "shard_file_5"
    input_dict = {"basename": basename, "shard": shard, "num_shards": num_shards, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6
    basename = np.array("validation_data", dtype=np.dtype('string'))
    shard = np.array(999, dtype=np.int32)
    num_shards = np.array(1000, dtype=np.int32)
    name = "shard_file_6"
    input_dict = {"basename": basename, "shard": shard, "num_shards": num_shards, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    basename = np.array("test_data", dtype=np.dtype('string'))
    shard = np.array(500, dtype=np.int32)
    num_shards = np.array(500, dtype=np.int32)
    name = "shard_file_7"
    input_dict = {"basename": basename, "shard": shard, "num_shards": num_shards, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    basename = np.array("very_long_file_name", dtype=np.dtype('string'))
    shard = np.array(10000, dtype=np.int32)
    num_shards = np.array(20000, dtype=np.int32)
    name = "shard_file_8"
    input_dict = {"basename": basename, "shard": shard, "num_shards": num_shards, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    basename = np.array("", dtype=np.dtype('string'))
    shard = np.array(0, dtype=np.int32)
    num_shards = np.array(1, dtype=np.int32)
    name = "shard_file_9"
    input_dict = {"basename": basename, "shard": shard, "num_shards": num_shards, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    basename = np.array("prefix", dtype=np.dtype('string'))
    shard = np.array(1, dtype=np.int32)
    num_shards = np.array(2, dtype=np.int32)
    name = "shard_file_10"
    input_dict = {"basename": basename, "shard": shard, "num_shards": num_shards, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ShardedFilename"] = tf_raw_ops_ShardedFilename_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ShardedFilename' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ShardedFilename'.")

check_valid('tf.raw_ops.ShardedFilename', generated_inputs['tf.raw_ops.ShardedFilename'], lib="tf", suffix=0)
