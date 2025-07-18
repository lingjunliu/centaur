
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import os
import tempfile
import tensorflow as tf

def get_tf_raw_ops_mergev2checkpoints_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.MergeV2Checkpoints function.
    This function creates dummy checkpoint files to satisfy the op's requirements, as
    the op interacts with the filesystem and needs at least one source file to exist.
    """
    list_of_inputs = []

    try:
        # Use a temporary directory for creating checkpoint files
        base_dir = os.environ.get("TEST_TMPDIR", tempfile.gettempdir())
        temp_dir_for_checkpoints = tempfile.mkdtemp(dir=base_dir)
    except (PermissionError, FileNotFoundError):
        temp_dir_for_checkpoints = tempfile.mkdtemp()

    def create_valid_dummy_checkpoint(prefix_path, var_suffix):
        """Creates a valid but minimal V2 checkpoint at the given prefix."""
        dir_name = os.path.dirname(prefix_path)
        os.makedirs(dir_name, exist_ok=True)
        # Each checkpoint needs a unique object graph to be mergeable.
        dummy_var = tf.Variable(0, name=f"dummy_var_{var_suffix}")
        # Also use a unique key for the object in the checkpoint
        ckpt = tf.train.Checkpoint(**{f"v_{var_suffix}": dummy_var})
        ckpt.write(prefix_path)

    # Case 1: Basic merge, one prefix exists, one is missing.
    case1_dir = os.path.join(temp_dir_for_checkpoints, 'case1')
    ckpt1_prefix_1 = os.path.join(case1_dir, 'ckpt1', 'model')
    ckpt2_prefix_1 = os.path.join(case1_dir, 'ckpt2', 'model') # This one won't be created
    dest_prefix_1 = os.path.join(case1_dir, 'merged', 'model')
    create_valid_dummy_checkpoint(ckpt1_prefix_1, "c1_v1")
    input_dict_1 = {
        'checkpoint_prefixes': np.array([ckpt1_prefix_1, ckpt2_prefix_1], dtype=object),
        'destination_prefix': np.array(dest_prefix_1, dtype=object),
        'delete_old_dirs': False,
        'allow_missing_files': True,
        'name': 'merge_one_missing'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Case 2: Both prefixes exist, allow_missing_files=False.
    case2_dir = os.path.join(temp_dir_for_checkpoints, 'case2')
    ckpt1_prefix_2 = os.path.join(case2_dir, 'shard1', 'ckpt')
    ckpt2_prefix_2 = os.path.join(case2_dir, 'shard2', 'ckpt')
    dest_prefix_2 = os.path.join(case2_dir, 'merged', 'ckpt')
    create_valid_dummy_checkpoint(ckpt1_prefix_2, "c2_v1")
    create_valid_dummy_checkpoint(ckpt2_prefix_2, "c2_v2")
    input_dict_2 = {
        'checkpoint_prefixes': np.array([ckpt1_prefix_2, ckpt2_prefix_2], dtype=object),
        'destination_prefix': np.array(dest_prefix_2, dtype=object),
        'delete_old_dirs': True,
        'allow_missing_files': False,
        'name': 'merge_and_delete'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Case 3: Merge into one of the source prefixes.
    case3_dir = os.path.join(temp_dir_for_checkpoints, 'case3')
    ckpt1_prefix_3 = os.path.join(case3_dir, 'v1', 'm')
    ckpt2_prefix_3 = os.path.join(case3_dir, 'v2', 'm')
    create_valid_dummy_checkpoint(ckpt1_prefix_3, "c3_v1")
    create_valid_dummy_checkpoint(ckpt2_prefix_3, "c3_v2")
    input_dict_3 = {
        'checkpoint_prefixes': np.array([ckpt1_prefix_3, ckpt2_prefix_3], dtype=object),
        'destination_prefix': np.array(ckpt1_prefix_3, dtype=object),
        'delete_old_dirs': True,
        'allow_missing_files': False,
        'name': 'merge_into_source'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Case 4: Single checkpoint "merge" (effectively a rename/move).
    case4_dir = os.path.join(temp_dir_for_checkpoints, 'case4')
    ckpt_prefix_4 = os.path.join(case4_dir, 'run_a', 'model')
    dest_prefix_4 = os.path.join(case4_dir, 'final', 'model')
    create_valid_dummy_checkpoint(ckpt_prefix_4, "c4_v1")
    input_dict_4 = {
        'checkpoint_prefixes': np.array([ckpt_prefix_4], dtype=object),
        'destination_prefix': np.array(dest_prefix_4, dtype=object),
        'delete_old_dirs': True,
        'allow_missing_files': False,
        'name': 'merge_single'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Case 5: Many prefixes, some missing, no name.
    case5_dir = os.path.join(temp_dir_for_checkpoints, 'case5')
    prefix_paths_5 = [os.path.join(case5_dir, f'p{i}') for i in range(5)]
    dest_prefix_5 = os.path.join(case5_dir, 'final_dest')
    create_valid_dummy_checkpoint(prefix_paths_5[1], "c5_v1")
    create_valid_dummy_checkpoint(prefix_paths_5[3], "c5_v2")
    input_dict_5 = {
        'checkpoint_prefixes': np.array(prefix_paths_5, dtype=object),
        'destination_prefix': np.array(dest_prefix_5, dtype=object),
        'delete_old_dirs': False,
        'allow_missing_files': True,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Case 6: Empty checkpoint_prefixes list.
    case6_dir = os.path.join(temp_dir_for_checkpoints, 'case6')
    dest_prefix_6 = os.path.join(case6_dir, 'dest')
    input_dict_6 = {
        'checkpoint_prefixes': np.array([], dtype=object),
        'destination_prefix': np.array(dest_prefix_6, dtype=object),
        'delete_old_dirs': False,
        'allow_missing_files': True,
        'name': 'merge_empty_list'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Case 7: Larger number of prefixes, all exist.
    case7_dir = os.path.join(temp_dir_for_checkpoints, 'case7')
    prefix_paths_7 = [os.path.join(case7_dir, f'shard_{i}') for i in range(4)]
    dest_prefix_7 = os.path.join(case7_dir, 'merged_large')
    for i, p in enumerate(prefix_paths_7):
        create_valid_dummy_checkpoint(p, f"c7_v{i}")
    input_dict_7 = {
        'checkpoint_prefixes': np.array(prefix_paths_7, dtype=object),
        'destination_prefix': np.array(dest_prefix_7, dtype=object),
        'delete_old_dirs': False,
        'allow_missing_files': False,
        'name': 'merge_many_existing'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Case 8: Destination is in a subdirectory of a source.
    case8_dir = os.path.join(temp_dir_for_checkpoints, 'case8')
    ckpt1_prefix_8 = os.path.join(case8_dir, 'src1', 'model')
    ckpt2_prefix_8 = os.path.join(case8_dir, 'src2', 'model')
    dest_prefix_8 = os.path.join(case8_dir, 'src1', 'sub', 'merged_model')
    create_valid_dummy_checkpoint(ckpt1_prefix_8, "c8_v1")
    create_valid_dummy_checkpoint(ckpt2_prefix_8, "c8_v2")
    input_dict_8 = {
        'checkpoint_prefixes': np.array([ckpt1_prefix_8, ckpt2_prefix_8], dtype=object),
        'destination_prefix': np.array(dest_prefix_8, dtype=object),
        'delete_old_dirs': False,
        'allow_missing_files': True,
        'name': 'merge_to_subdir'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Case 9: Prefixes with dots in the name.
    case9_dir = os.path.join(temp_dir_for_checkpoints, 'case9')
    ckpt1_prefix_9 = os.path.join(case9_dir, 'model.part-1')
    ckpt2_prefix_9 = os.path.join(case9_dir, 'model.part-2')
    dest_prefix_9 = os.path.join(case9_dir, 'model.final')
    create_valid_dummy_checkpoint(ckpt1_prefix_9, "c9_v1")
    create_valid_dummy_checkpoint(ckpt2_prefix_9, "c9_v2")
    input_dict_9 = {
        'checkpoint_prefixes': np.array([ckpt1_prefix_9, ckpt2_prefix_9], dtype=object),
        'destination_prefix': np.array(dest_prefix_9, dtype=object),
        'delete_old_dirs': False,
        'allow_missing_files': False,
        'name': 'merge_with_dots'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Case 10: Long file paths.
    case10_dir = os.path.join(temp_dir_for_checkpoints, 'a' * 15, 'b' * 15, 'c' * 15)
    ckpt1_prefix_10 = os.path.join(case10_dir, 'long_path_ckpt')
    dest_prefix_10 = os.path.join(case10_dir, 'long_path_merged')
    create_valid_dummy_checkpoint(ckpt1_prefix_10, "c10_v1")
    input_dict_10 = {
        'checkpoint_prefixes': np.array([ckpt1_prefix_10], dtype=object),
        'destination_prefix': np.array(dest_prefix_10, dtype=object),
        'delete_old_dirs': False,
        'allow_missing_files': True,
        'name': 'long_path_merge'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.MergeV2Checkpoints"] = get_tf_raw_ops_mergev2checkpoints_inputs()

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
