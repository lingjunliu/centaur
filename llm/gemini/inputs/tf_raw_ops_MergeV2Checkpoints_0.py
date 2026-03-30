
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_MergeV2Checkpoints_inputs():
    list_of_inputs = []

    # Input 1
    checkpoint_prefixes = np.array(["/tmp/model1", "/tmp/model2"], dtype=np.object_)
    destination_prefix = np.array("/tmp/merged_model", dtype=np.object_)
    delete_old_dirs = np.array(True, dtype=np.bool_)
    allow_missing_files = np.array(True, dtype=np.bool_)
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
    checkpoint_prefixes = np.array(["/tmp/model_a", "/tmp/model_b", "/tmp/model_c"], dtype=np.object_)
    destination_prefix = np.array("/tmp/final_model", dtype=np.object_)
    delete_old_dirs = np.array(False, dtype=np.bool_)
    allow_missing_files = np.array(True, dtype=np.bool_)
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
    checkpoint_prefixes = np.array(["/tmp/shard1", "/tmp/shard2"], dtype=np.object_)
    destination_prefix = np.array("/tmp/shard1", dtype=np.object_)
    delete_old_dirs = np.array(True, dtype=np.bool_)
    allow_missing_files = np.array(True, dtype=np.bool_)
    name = "merge_checkpoints_3"

    input_dict = {
        "checkpoint_prefixes": checkpoint_prefixes,
        "destination_prefix": destination_prefix,
        "delete_old_dirs": delete_old_dirs,
        "allow_missing_files": allow_missing_files,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    checkpoint_prefixes = np.array([], dtype=np.object_)
    destination_prefix = np.array("/tmp/empty_merge", dtype=np.object_)
    delete_old_dirs = np.array(False, dtype=np.bool_)
    allow_missing_files = np.array(True, dtype=np.bool_)
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
    checkpoint_prefixes = np.array(["/tmp/model_x"], dtype=np.object_)
    destination_prefix = np.array("/tmp/model_y", dtype=np.object_)
    delete_old_dirs = np.array(True, dtype=np.bool_)
    allow_missing_files = np.array(True, dtype=np.bool_)
    name = None

    input_dict = {
        "checkpoint_prefixes": checkpoint_prefixes,
        "destination_prefix": destination_prefix,
        "delete_old_dirs": delete_old_dirs,
        "allow_missing_files": allow_missing_files,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6
    checkpoint_prefixes = np.array(["/tmp/model_p", "/tmp/model_q", "/tmp/model_r", "/tmp/model_s"], dtype=np.object_)
    destination_prefix = np.array("/tmp/super_model", dtype=np.object_)
    delete_old_dirs = np.array(True, dtype=np.bool_)
    allow_missing_files = np.array(True, dtype=np.bool_)
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
    checkpoint_prefixes = np.array(["/tmp/a/b/c", "/tmp/a/b/d"], dtype=np.object_)
    destination_prefix = np.array("/tmp/merged_a", dtype=np.object_)
    delete_old_dirs = np.array(False, dtype=np.bool_)
    allow_missing_files = np.array(True, dtype=np.bool_)
    name = "merge_checkpoints_7"

    input_dict = {
        "checkpoint_prefixes": checkpoint_prefixes,
        "destination_prefix": destination_prefix,
        "delete_old_dirs": delete_old_dirs,
        "allow_missing_files": allow_missing_files,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    checkpoint_prefixes = np.array(["/tmp/one_shard"], dtype=np.object_)
    destination_prefix = np.array("/tmp/the_result", dtype=np.object_)
    delete_old_dirs = np.array(True, dtype=np.bool_)
    allow_missing_files = np.array(True, dtype=np.bool_)
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
    checkpoint_prefixes = np.array(["/tmp/m1", "/tmp/m2", "/tmp/m3"], dtype=np.object_)
    destination_prefix = np.array("/tmp/m_all", dtype=np.object_)
    delete_old_dirs = np.array(False, dtype=np.bool_)
    allow_missing_files = np.array(True, dtype=np.bool_)
    name = None

    input_dict = {
        "checkpoint_prefixes": checkpoint_prefixes,
        "destination_prefix": destination_prefix,
        "delete_old_dirs": delete_old_dirs,
        "allow_missing_files": allow_missing_files,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    checkpoint_prefixes = np.array(["/tmp/c1", "/tmp/c2"], dtype=np.object_)
    destination_prefix = np.array("/tmp/c1", dtype=np.object_)
    delete_old_dirs = np.array(True, dtype=np.bool_)
    allow_missing_files = np.array(True, dtype=np.bool_)
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
generated_inputs["tf.raw_ops.MergeV2Checkpoints"] = tf_raw_ops_MergeV2Checkpoints_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.MergeV2Checkpoints' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MergeV2Checkpoints'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.MergeV2Checkpoints', generated_inputs['tf.raw_ops.MergeV2Checkpoints'], lib="tf", suffix=0)
