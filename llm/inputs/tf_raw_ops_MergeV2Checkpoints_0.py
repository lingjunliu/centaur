
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_merge_v2_checkpoints_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.MergeV2Checkpoints function.
    """
    list_of_inputs = []

    # Input 1: Basic case, allow_missing_files set to True to prevent runtime error
    input_dict_1 = {
        'checkpoint_prefixes': np.array(['/tmp/ckpt_shard1/ckpt', '/tmp/ckpt_shard2/ckpt'], dtype=np.object_),
        'destination_prefix': np.array('/tmp/merged_ckpt/ckpt', dtype=np.object_),
        'delete_old_dirs': True,
        'allow_missing_files': True,
        'name': 'merge_op_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Do not delete old directories
    input_dict_2 = {
        'checkpoint_prefixes': np.array(['./model_a', './model_b'], dtype=np.object_),
        'destination_prefix': np.array('./model_merged', dtype=np.object_),
        'delete_old_dirs': False,
        'allow_missing_files': True,
        'name': 'merge_no_delete'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Allow missing files (original)
    input_dict_3 = {
        'checkpoint_prefixes': np.array(['run1/chkpt', 'run2/chkpt', 'run3/chkpt'], dtype=np.object_),
        'destination_prefix': np.array('final_run/chkpt', dtype=np.object_),
        'delete_old_dirs': True,
        'allow_missing_files': True,
        'name': 'tolerant_merge'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Both optional bools set to non-default values
    input_dict_4 = {
        'checkpoint_prefixes': np.array(['/data/shard_x', '/data/shard_y'], dtype=np.object_),
        'destination_prefix': np.array('/data/final_x_y', dtype=np.object_),
        'delete_old_dirs': False,
        'allow_missing_files': True,
        'name': 'merge_and_clean_tolerant'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))
    
    # Input 5: Both optional bools set to their default values explicitly
    input_dict_5 = {
        'checkpoint_prefixes': np.array(['c:/windows/temp/a', 'c:/windows/temp/b'], dtype=np.object_),
        'destination_prefix': np.array('c:/windows/temp/merged', dtype=np.object_),
        'delete_old_dirs': True,
        'allow_missing_files': True,
        'name': 'strict_merge_with_delete'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Destination is the same as one of the source prefixes
    input_dict_6 = {
        'checkpoint_prefixes': np.array(['ckpt-part-1', 'ckpt-part-2', 'ckpt-part-3'], dtype=np.object_),
        'destination_prefix': np.array('ckpt-part-1', dtype=np.object_),
        'delete_old_dirs': True,
        'allow_missing_files': True,
        'name': 'merge_in_place'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Larger number of checkpoint prefixes
    input_dict_7 = {
        'checkpoint_prefixes': np.array(['s1', 's2', 's3', 's4', 's5'], dtype=np.object_),
        'destination_prefix': np.array('merged_all', dtype=np.object_),
        'delete_old_dirs': False,
        'allow_missing_files': True,
        'name': 'merge_many'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Single source prefix (edge case)
    input_dict_8 = {
        'checkpoint_prefixes': np.array(['only_one_ckpt/model'], dtype=np.object_),
        'destination_prefix': np.array('moved_ckpt/model', dtype=np.object_),
        'delete_old_dirs': True,
        'allow_missing_files': True,
        'name': 'merge_single'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: No optional name provided
    input_dict_9 = {
        'checkpoint_prefixes': np.array(['/path/to/shard/0', '/path/to/shard/1'], dtype=np.object_),
        'destination_prefix': np.array('/path/to/final_ckpt', dtype=np.object_),
        'delete_old_dirs': False,
        'allow_missing_files': True,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))
    
    # Input 10: Complex relative paths
    input_dict_10 = {
        'checkpoint_prefixes': np.array(['../backups/2023-10-26/ckpt.v2', './temp/run_123/ckpt.v2'], dtype=np.object_),
        'destination_prefix': np.array('./final/production_model/ckpt.v2', dtype=np.object_),
        'delete_old_dirs': False,
        'allow_missing_files': True,
        'name': 'complex_path_merge'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.MergeV2Checkpoints"] = tf_raw_ops_merge_v2_checkpoints_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.MergeV2Checkpoints' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MergeV2Checkpoints'.")

check_valid('tf.raw_ops.MergeV2Checkpoints', generated_inputs['tf.raw_ops.MergeV2Checkpoints'], lib="tf", suffix=0)
