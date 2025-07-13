
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_merge_v2_checkpoints_inputs():
    list_of_inputs = []

    # Input 1
    checkpoint_prefixes = np.array(["/tmp/model.ckpt-100", "/tmp/model.ckpt-200"], dtype='|S256')
    destination_prefix = np.array("/tmp/merged_model.ckpt", dtype='|S256')
    delete_old_dirs = True
    allow_missing_files = False
    name = "merge_checkpoints_1"
    input_dict = {
        "checkpoint_prefixes": checkpoint_prefixes,
        "destination_prefix": destination_prefix,
        "delete_old_dirs": delete_old_dirs,
        "allow_missing_files": allow_missing_files,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    checkpoint_prefixes = np.array(["/tmp/model.ckpt-100"], dtype='|S256')
    destination_prefix = np.array("/tmp/merged_model.ckpt", dtype='|S256')
    delete_old_dirs = False
    allow_missing_files = True
    name = "merge_checkpoints_2"
    input_dict = {
        "checkpoint_prefixes": checkpoint_prefixes,
        "destination_prefix": destination_prefix,
        "delete_old_dirs": delete_old_dirs,
        "allow_missing_files": allow_missing_files,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    checkpoint_prefixes = np.array(["/tmp/model.ckpt-100", "/tmp/model.ckpt-200", "/tmp/model.ckpt-300"], dtype='|S256')
    destination_prefix = np.array("/tmp/merged_model.ckpt", dtype='|S256')
    delete_old_dirs = True
    allow_missing_files = False
    name = None
    input_dict = {
        "checkpoint_prefixes": checkpoint_prefixes,
        "destination_prefix": destination_prefix,
        "delete_old_dirs": delete_old_dirs,
        "allow_missing_files": allow_missing_files,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    checkpoint_prefixes = np.array(["/tmp/model.ckpt-100"], dtype='|S256')
    destination_prefix = np.array("/tmp/model.ckpt-100", dtype='|S256')
    delete_old_dirs = False
    allow_missing_files = False
    name = "merge_checkpoints_4"
    input_dict = {
        "checkpoint_prefixes": checkpoint_prefixes,
        "destination_prefix": destination_prefix,
        "delete_old_dirs": delete_old_dirs,
        "allow_missing_files": allow_missing_files,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    checkpoint_prefixes = np.array([], dtype='|S256')
    destination_prefix = np.array("/tmp/merged_model.ckpt", dtype='|S256')
    delete_old_dirs = True
    allow_missing_files = True
    name = "merge_checkpoints_5"
    input_dict = {
        "checkpoint_prefixes": checkpoint_prefixes,
        "destination_prefix": destination_prefix,
        "delete_old_dirs": delete_old_dirs,
        "allow_missing_files": allow_missing_files,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    checkpoint_prefixes = np.array(["/tmp/model.ckpt-abc"], dtype='|S256')
    destination_prefix = np.array("/tmp/merged_model.ckpt", dtype='|S256')
    delete_old_dirs = True
    allow_missing_files = True
    name = "merge_checkpoints_6"
    input_dict = {
        "checkpoint_prefixes": checkpoint_prefixes,
        "destination_prefix": destination_prefix,
        "delete_old_dirs": delete_old_dirs,
        "allow_missing_files": allow_missing_files,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    checkpoint_prefixes = np.array(["/tmp/model.ckpt-1", "/tmp/model.ckpt-2"], dtype='|S256')
    destination_prefix = np.array("/tmp/merged_model.ckpt", dtype='|S256')
    delete_old_dirs = False
    allow_missing_files = False
    name = None
    input_dict = {
        "checkpoint_prefixes": checkpoint_prefixes,
        "destination_prefix": destination_prefix,
        "delete_old_dirs": delete_old_dirs,
        "allow_missing_files": allow_missing_files,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    checkpoint_prefixes = np.array(["/tmp/model.ckpt-100", "/tmp/model.ckpt-200"], dtype='|S256')
    destination_prefix = np.array("/tmp/merged_model.ckpt", dtype='|S256')
    delete_old_dirs = True
    allow_missing_files = True
    name = "merge_checkpoints_8"
    input_dict = {
        "checkpoint_prefixes": checkpoint_prefixes,
        "destination_prefix": destination_prefix,
        "delete_old_dirs": delete_old_dirs,
        "allow_missing_files": allow_missing_files,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    checkpoint_prefixes = np.array(["/tmp/model.ckpt-100"], dtype='|S256')
    destination_prefix = np.array("/tmp/merged_model.ckpt", dtype='|S256')
    delete_old_dirs = True
    allow_missing_files = False
    name = "merge_checkpoints_9"
    input_dict = {
        "checkpoint_prefixes": checkpoint_prefixes,
        "destination_prefix": destination_prefix,
        "delete_old_dirs": delete_old_dirs,
        "allow_missing_files": allow_missing_files,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    checkpoint_prefixes = np.array(["/tmp/a/model.ckpt-100", "/tmp/b/model.ckpt-200"], dtype='|S256')
    destination_prefix = np.array("/tmp/merged_model.ckpt", dtype='|S256')
    delete_old_dirs = False
    allow_missing_files = True
    name = "merge_checkpoints_10"
    input_dict = {
        "checkpoint_prefixes": checkpoint_prefixes,
        "destination_prefix": destination_prefix,
        "delete_old_dirs": delete_old_dirs,
        "allow_missing_files": allow_missing_files,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.MergeV2Checkpoints"] = tf_raw_ops_merge_v2_checkpoints_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.MergeV2Checkpoints' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MergeV2Checkpoints'.")

check_valid('tf.raw_ops.MergeV2Checkpoints', generated_inputs['tf.raw_ops.MergeV2Checkpoints'], lib="tf", suffix=0)
