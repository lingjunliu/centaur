
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import os

def tf_train_get_checkpoint_state_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    checkpoint_dir = "./checkpoint_dir_1"
    latest_filename = "checkpoint"
    input_dict = {"checkpoint_dir": checkpoint_dir, "latest_filename": latest_filename}
    list_of_inputs.append(copy.deepcopy(input_dict))
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, latest_filename), "w") as f:
        f.write("model_checkpoint_path: \"model.ckpt-1000\"\nall_model_checkpoint_paths: \"model.ckpt-1000\"")
    
    # Input 2: Different checkpoint filename
    checkpoint_dir = "./checkpoint_dir_2"
    latest_filename = "my_checkpoint"
    input_dict = {"checkpoint_dir": checkpoint_dir, "latest_filename": latest_filename}
    list_of_inputs.append(copy.deepcopy(input_dict))
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, latest_filename), "w") as f:
        f.write("model_checkpoint_path: \"model.ckpt-2000\"\nall_model_checkpoint_paths: \"model.ckpt-2000\"")

    # Input 3:  Empty checkpoint directory (should return None, but is a valid input)
    checkpoint_dir = "./checkpoint_dir_3"
    latest_filename = "checkpoint"
    input_dict = {"checkpoint_dir": checkpoint_dir, "latest_filename": latest_filename}
    list_of_inputs.append(copy.deepcopy(input_dict))
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Input 4: checkpoint dir with no checkpoint file, but other files
    checkpoint_dir = "./checkpoint_dir_4"
    latest_filename = "checkpoint"
    input_dict = {"checkpoint_dir": checkpoint_dir, "latest_filename": latest_filename}
    list_of_inputs.append(copy.deepcopy(input_dict))
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, "other_file.txt"), "w") as f:
        f.write("Some data")
    
    # Input 5: Checkpoint file with multiple checkpoint paths
    checkpoint_dir = "./checkpoint_dir_5"
    latest_filename = "checkpoint"
    input_dict = {"checkpoint_dir": checkpoint_dir, "latest_filename": latest_filename}
    list_of_inputs.append(copy.deepcopy(input_dict))
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, latest_filename), "w") as f:
        f.write("model_checkpoint_path: \"model.ckpt-3000\"\nall_model_checkpoint_paths: \"model.ckpt-3000\"\nall_model_checkpoint_paths: \"model.ckpt-3001\"")

    # Input 6: checkpoint dir with subdirectories
    checkpoint_dir = "./checkpoint_dir_6/subdir"
    latest_filename = "checkpoint"
    input_dict = {"checkpoint_dir": checkpoint_dir, "latest_filename": latest_filename}
    list_of_inputs.append(copy.deepcopy(input_dict))
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, latest_filename), "w") as f:
        f.write("model_checkpoint_path: \"model.ckpt-4000\"\nall_model_checkpoint_paths: \"model.ckpt-4000\"")

    # Input 7: different latest_filename, same dir
    checkpoint_dir = "./checkpoint_dir_7"
    latest_filename = "latest"
    input_dict = {"checkpoint_dir": checkpoint_dir, "latest_filename": latest_filename}
    list_of_inputs.append(copy.deepcopy(input_dict))
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, latest_filename), "w") as f:
        f.write("model_checkpoint_path: \"model.ckpt-5000\"\nall_model_checkpoint_paths: \"model.ckpt-5000\"")

    # Input 8:  checkpoint with no all_model_checkpoint_paths entry
    checkpoint_dir = "./checkpoint_dir_8"
    latest_filename = "checkpoint"
    input_dict = {"checkpoint_dir": checkpoint_dir, "latest_filename": latest_filename}
    list_of_inputs.append(copy.deepcopy(input_dict))
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, latest_filename), "w") as f:
        f.write("model_checkpoint_path: \"model.ckpt-6000\"")

    # Input 10: checkpoint with long model path name
    checkpoint_dir = "./checkpoint_dir_10"
    latest_filename = "checkpoint"
    model_name = "model_with_very_very_long_name_that_should_still_work.ckpt-7000"
    input_dict = {"checkpoint_dir": checkpoint_dir, "latest_filename": latest_filename}
    list_of_inputs.append(copy.deepcopy(input_dict))
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, latest_filename), "w") as f:
        f.write(f"model_checkpoint_path: \"{model_name}\"\nall_model_checkpoint_paths: \"{model_name}\"")
        
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.train.get_checkpoint_state"] = tf_train_get_checkpoint_state_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.train.get_checkpoint_state' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.train.get_checkpoint_state'.")

check_valid('tf.train.get_checkpoint_state', generated_inputs['tf.train.get_checkpoint_state'], lib="tf", suffix=0)
