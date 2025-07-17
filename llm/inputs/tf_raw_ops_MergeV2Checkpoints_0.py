
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
import os
import tempfile

def tf_raw_ops_MergeV2Checkpoints_inputs():
    list_of_inputs = []

    # Input 1
    checkpoint_prefixes = np.array([os.path.join(tempfile.gettempdir(), "model1", "ckpt-1")], dtype=np.object_)
    destination_prefix = np.array(os.path.join(tempfile.gettempdir(), "merged_model"), dtype=np.object_)
    delete_old_dirs = True
    allow_missing_files = True
    name = "merge_checkpoints_1"
    
    # Create dummy file
    if not os.path.exists(os.path.join(tempfile.gettempdir(), "model1")):
        os.makedirs(os.path.join(tempfile.gettempdir(), "model1"), exist_ok=True)
    with open(os.path.join(tempfile.gettempdir(), "model1", "ckpt-1"), "w") as f:
        f.write("dummy data")


    input_dict = {
        "checkpoint_prefixes": checkpoint_prefixes,
        "destination_prefix": destination_prefix,
        "delete_old_dirs": delete_old_dirs,
        "allow_missing_files": allow_missing_files,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    checkpoint_prefixes = np.array([os.path.join(tempfile.gettempdir(), "model2", "ckpt-1")], dtype=np.object_)
    destination_prefix = np.array(os.path.join(tempfile.gettempdir(), "model2", "ckpt-merged"), dtype=np.object_)
    delete_old_dirs = False
    allow_missing_files = True
    name = "merge_checkpoints_2"

    # Create dummy file
    if not os.path.exists(os.path.join(tempfile.gettempdir(), "model2")):
        os.makedirs(os.path.join(tempfile.gettempdir(), "model2"), exist_ok=True)
    with open(os.path.join(tempfile.gettempdir(), "model2", "ckpt-1"), "w") as f:
        f.write("dummy data")


    input_dict = {
        "checkpoint_prefixes": checkpoint_prefixes,
        "destination_prefix": destination_prefix,
        "delete_old_dirs": delete_old_dirs,
        "allow_missing_files": allow_missing_files,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    checkpoint_prefixes = np.array([os.path.join(tempfile.gettempdir(), "model3", "ckpt-a"), os.path.join(tempfile.gettempdir(), "model3", "ckpt-b")], dtype=np.object_)
    destination_prefix = np.array(os.path.join(tempfile.gettempdir(), "model3", "merged"), dtype=np.object_)
    delete_old_dirs = True
    allow_missing_files = True
    name = "merge_checkpoints_3"

    # Create dummy files
    if not os.path.exists(os.path.join(tempfile.gettempdir(), "model3")):
        os.makedirs(os.path.join(tempfile.gettempdir(), "model3"), exist_ok=True)
    with open(os.path.join(tempfile.gettempdir(), "model3", "ckpt-a"), "w") as f:
        f.write("dummy data")
    with open(os.path.join(tempfile.gettempdir(), "model3", "ckpt-b"), "w") as f:
        f.write("dummy data")

    input_dict = {
        "checkpoint_prefixes": checkpoint_prefixes,
        "destination_prefix": destination_prefix,
        "delete_old_dirs": delete_old_dirs,
        "allow_missing_files": allow_missing_files,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    checkpoint_prefixes = np.array([os.path.join(tempfile.gettempdir(), "model4", "shard1")], dtype=np.object_)
    destination_prefix = np.array(os.path.join(tempfile.gettempdir(), "model4", "final"), dtype=np.object_)
    delete_old_dirs = False
    allow_missing_files = False
    name = "merge_checkpoints_4"

    # Create dummy files
    if not os.path.exists(os.path.join(tempfile.gettempdir(), "model4")):
        os.makedirs(os.path.join(tempfile.gettempdir(), "model4"), exist_ok=True)
    with open(os.path.join(tempfile.gettempdir(), "model4", "shard1"), "w") as f:
        f.write("dummy data")

    input_dict = {
        "checkpoint_prefixes": checkpoint_prefixes,
        "destination_prefix": destination_prefix,
        "delete_old_dirs": delete_old_dirs,
        "allow_missing_files": allow_missing_files,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    checkpoint_prefixes = np.array([os.path.join(tempfile.gettempdir(), "ckpt_part_1")], dtype=np.object_)
    destination_prefix = np.array(os.path.join(tempfile.gettempdir(), "merged_ckpt"), dtype=np.object_)
    delete_old_dirs = True
    allow_missing_files = True
    name = None

    # Create dummy file
    if not os.path.exists(tempfile.gettempdir()):
        os.makedirs(tempfile.gettempdir(), exist_ok=True)
    with open(os.path.join(tempfile.gettempdir(), "ckpt_part_1"), "w") as f:
        f.write("dummy data")

    input_dict = {
        "checkpoint_prefixes": checkpoint_prefixes,
        "destination_prefix": destination_prefix,
        "delete_old_dirs": delete_old_dirs,
        "allow_missing_files": allow_missing_files,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    checkpoint_prefixes = np.array([os.path.join(tempfile.gettempdir(), "ckpt1")], dtype=np.object_)
    destination_prefix = np.array(os.path.join(tempfile.gettempdir(), "merged_ckpt"), dtype=np.object_)
    delete_old_dirs = False
    allow_missing_files = True
    name = ""

    # Create dummy files
    if not os.path.exists(tempfile.gettempdir()):
        os.makedirs(tempfile.gettempdir(), exist_ok=True)
    with open(os.path.join(tempfile.gettempdir(), "ckpt1"), "w") as f:
        f.write("dummy data")

    input_dict = {
        "checkpoint_prefixes": checkpoint_prefixes,
        "destination_prefix": destination_prefix,
        "delete_old_dirs": delete_old_dirs,
        "allow_missing_files": allow_missing_files,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7
    checkpoint_prefixes = np.array([os.path.join(tempfile.gettempdir(), "model7", "ckpt-1"), os.path.join(tempfile.gettempdir(), "model7", "ckpt-2")], dtype=np.object_)
    destination_prefix = np.array(os.path.join(tempfile.gettempdir(), "model7", "ckpt-1"), dtype=np.object_)
    delete_old_dirs = True
    allow_missing_files = True
    name = "merge_checkpoints_7"
    
    # Create dummy files
    if not os.path.exists(os.path.join(tempfile.gettempdir(), "model7")):
        os.makedirs(os.path.join(tempfile.gettempdir(), "model7"), exist_ok=True)
    with open(os.path.join(tempfile.gettempdir(), "model7", "ckpt-1"), "w") as f:
        f.write("dummy data")
    with open(os.path.join(tempfile.gettempdir(), "model7", "ckpt-2"), "w") as f:
        f.write("dummy data")

    input_dict = {
        "checkpoint_prefixes": checkpoint_prefixes,
        "destination_prefix": destination_prefix,
        "delete_old_dirs": delete_old_dirs,
        "allow_missing_files": allow_missing_files,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    checkpoint_prefixes = np.array([os.path.join(tempfile.gettempdir(), "model8", "ckpt-1")], dtype=np.object_)
    destination_prefix = np.array(os.path.join(tempfile.gettempdir(), "merged_model"), dtype=np.object_)
    delete_old_dirs = True
    allow_missing_files = True
    name = "merge_checkpoints_8"

    # Create dummy file
    if not os.path.exists(os.path.join(tempfile.gettempdir(), "model8")):
        os.makedirs(os.path.join(tempfile.gettempdir(), "model8"), exist_ok=True)
    with open(os.path.join(tempfile.gettempdir(), "model8", "ckpt-1"), "w") as f:
        f.write("dummy data")

    input_dict = {
        "checkpoint_prefixes": checkpoint_prefixes,
        "destination_prefix": destination_prefix,
        "delete_old_dirs": delete_old_dirs,
        "allow_missing_files": allow_missing_files,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    checkpoint_prefixes = np.array([os.path.join(tempfile.gettempdir(), "model9", "ckpt-1"), os.path.join(tempfile.gettempdir(), "model9", "ckpt-2"), os.path.join(tempfile.gettempdir(), "model9", "ckpt-3")], dtype=np.object_)
    destination_prefix = np.array(os.path.join(tempfile.gettempdir(), "merged_model"), dtype=np.object_)
    delete_old_dirs = False
    allow_missing_files = True
    name = "merge_checkpoints_9"

    # Create dummy files
    if not os.path.exists(os.path.join(tempfile.gettempdir(), "model9")):
        os.makedirs(os.path.join(tempfile.gettempdir(), "model9"), exist_ok=True)
    with open(os.path.join(tempfile.gettempdir(), "model9", "ckpt-1"), "w") as f:
        f.write("dummy data")
    with open(os.path.join(tempfile.gettempdir(), "model9", "ckpt-2"), "w") as f:
        f.write("dummy data")
    with open(os.path.join(tempfile.gettempdir(), "model9", "ckpt-3"), "w") as f:
        f.write("dummy data")

    input_dict = {
        "checkpoint_prefixes": checkpoint_prefixes,
        "destination_prefix": destination_prefix,
        "delete_old_dirs": delete_old_dirs,
        "allow_missing_files": allow_missing_files,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    checkpoint_prefixes = np.array([os.path.join(tempfile.gettempdir(), "model10", "ckpt-1"), os.path.join(tempfile.gettempdir(), "model10", "ckpt-2")], dtype=np.object_)
    destination_prefix = np.array(os.path.join(tempfile.gettempdir(), "merged_model"), dtype=np.object_)
    delete_old_dirs = True
    allow_missing_files = True
    name = "merge_checkpoints_10"
    
    # Create dummy files
    if not os.path.exists(os.path.join(tempfile.gettempdir(), "model10")):
        os.makedirs(os.path.join(tempfile.gettempdir(), "model10"), exist_ok=True)
    with open(os.path.join(tempfile.gettempdir(), "model10", "ckpt-1"), "w") as f:
        f.write("dummy data")
    with open(os.path.join(tempfile.gettempdir(), "model10", "ckpt-2"), "w") as f:
        f.write("dummy data")


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
    
    print("Valid")

if 'tf.raw_ops.MergeV2Checkpoints' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MergeV2Checkpoints'.")

check_valid('tf.raw_ops.MergeV2Checkpoints', generated_inputs['tf.raw_ops.MergeV2Checkpoints'], lib="tf", suffix=0)
