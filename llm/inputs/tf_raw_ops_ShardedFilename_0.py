
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_shardedfilename_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.ShardedFilename.
    """
    list_of_inputs = []

    # Input 1: Basic case
    input_dict = {
        'basename': np.array('/path/to/file', dtype=object),
        'shard': np.array(0, dtype=np.int32),
        'num_shards': np.array(10, dtype=np.int32),
        'name': 'basic_sharding'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different shard and num_shards
    input_dict = {
        'basename': np.array('data/train', dtype=object),
        'shard': np.array(5, dtype=np.int32),
        'num_shards': np.array(100, dtype=np.int32),
        'name': 'train_data_sharding'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Max shard value (shard = num_shards - 1)
    input_dict = {
        'basename': np.array('output_file', dtype=object),
        'shard': np.array(99, dtype=np.int32),
        'num_shards': np.array(100, dtype=np.int32),
        'name': 'max_shard'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Single shard
    input_dict = {
        'basename': np.array('single_shard_file.dat', dtype=object),
        'shard': np.array(0, dtype=np.int32),
        'num_shards': np.array(1, dtype=np.int32),
        'name': 'single_shard'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: No name provided
    input_dict = {
        'basename': np.array('another-file', dtype=object),
        'shard': np.array(123, dtype=np.int32),
        'num_shards': np.array(1000, dtype=np.int32),
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Basename with dots (like a file extension)
    input_dict = {
        'basename': np.array('model.ckpt', dtype=object),
        'shard': np.array(4, dtype=np.int32),
        'num_shards': np.array(5, dtype=np.int32),
        'name': 'checkpoint_sharding'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Long basename
    input_dict = {
        'basename': np.array('a_very_long_and_descriptive_filename_for_a_specific_dataset_split', dtype=object),
        'shard': np.array(0, dtype=np.int32),
        'num_shards': np.array(2, dtype=np.int32),
        'name': 'long_name_test'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty basename string
    input_dict = {
        'basename': np.array('', dtype=object),
        'shard': np.array(10, dtype=np.int32),
        'num_shards': np.array(20, dtype=np.int32),
        'name': 'empty_basename'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large shard and num_shards values (equal to padding)
    input_dict = {
        'basename': np.array('large_shards_data', dtype=object),
        'shard': np.array(9999, dtype=np.int32),
        'num_shards': np.array(10000, dtype=np.int32),
        'name': 'large_numbers_test'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Basename with underscores
    input_dict = {
        'basename': np.array('file_with_underscore', dtype=object),
        'shard': np.array(7, dtype=np.int32),
        'num_shards': np.array(8, dtype=np.int32),
        'name': 'underscore_test'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Shard > num_shards (logically inconsistent but valid for the op)
    input_dict = {
        'basename': np.array('invalid_shard_test', dtype=object),
        'shard': np.array(10, dtype=np.int32),
        'num_shards': np.array(5, dtype=np.int32),
        'name': 'shard_greater_than_total'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: Numbers larger than 5-digit padding
    input_dict = {
        'basename': np.array('six_digit_shard', dtype=object),
        'shard': np.array(123456, dtype=np.int32),
        'num_shards': np.array(200000, dtype=np.int32),
        'name': 'six_digit_test'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.ShardedFilename"] = tf_raw_ops_shardedfilename_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ShardedFilename' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ShardedFilename'.")

check_valid('tf.raw_ops.ShardedFilename', generated_inputs['tf.raw_ops.ShardedFilename'], lib="tf", suffix=0)
