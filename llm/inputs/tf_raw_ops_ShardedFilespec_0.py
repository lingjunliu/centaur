
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_shardedfilespec_inputs():
    list_of_inputs = []

    # Input 1
    basename = np.array("data", dtype=np.string_)
    num_shards = np.array(5, dtype=np.int32)
    name = "sharded_file_1"

    input_dict = {
        "basename": basename,
        "num_shards": num_shards,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    basename = np.array("train.tfrecord", dtype=np.string_)
    num_shards = np.array(10, dtype=np.int32)
    name = None

    input_dict = {
        "basename": basename,
        "num_shards": num_shards,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    basename = np.array("validation", dtype=np.string_)
    num_shards = np.array(1, dtype=np.int32)
    name = "single_shard"

    input_dict = {
        "basename": basename,
        "num_shards": num_shards,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    basename = np.array("test_data", dtype=np.string_)
    num_shards = np.array(20, dtype=np.int32)
    name = "many_shards"

    input_dict = {
        "basename": basename,
        "num_shards": num_shards,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    basename = np.array("model", dtype=np.string_)
    num_shards = np.array(2, dtype=np.int32)
    name = "model_shards"

    input_dict = {
        "basename": basename,
        "num_shards": num_shards,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    basename = np.array("image_dataset", dtype=np.string_)
    num_shards = np.array(100, dtype=np.int32)
    name = "image_shards"

    input_dict = {
        "basename": basename,
        "num_shards": num_shards,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    basename = np.array("audio_data", dtype=np.string_)
    num_shards = np.array(8, dtype=np.int32)
    name = "audio_shards"

    input_dict = {
        "basename": basename,
        "num_shards": num_shards,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    basename = np.array("checkpoint", dtype=np.string_)
    num_shards = np.array(3, dtype=np.int32)
    name = "checkpoint_shards"

    input_dict = {
        "basename": basename,
        "num_shards": num_shards,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    basename = np.array("embeddings", dtype=np.string_)
    num_shards = np.array(64, dtype=np.int32)
    name = "embedding_shards"

    input_dict = {
        "basename": basename,
        "num_shards": num_shards,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    basename = np.array("logs", dtype=np.string_)
    num_shards = np.array(4, dtype=np.int32)
    name = "log_shards"

    input_dict = {
        "basename": basename,
        "num_shards": num_shards,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ShardedFilespec' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ShardedFilespec'.")

check_valid('tf.raw_ops.ShardedFilespec', generated_inputs['tf.raw_ops.ShardedFilespec'], lib="tf", suffix=0)
