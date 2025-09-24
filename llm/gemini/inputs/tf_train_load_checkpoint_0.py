
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import os

def tf_train_load_checkpoint_inputs():
    list_of_inputs = []

    # Helper function to create a checkpoint file
    def create_checkpoint(ckpt_dir, var_name, var_value):
        os.makedirs(ckpt_dir, exist_ok=True)
        a = tf.Variable(1.0)
        b = tf.Variable(2.0)
        ckpt = tf.train.Checkpoint(var_list={'a': a, 'b': b})
        ckpt_path = ckpt.save(os.path.join(ckpt_dir, 'tmp-ckpt'))
        return ckpt_path
        
    # Input 1: Valid checkpoint path
    ckpt_dir = "checkpoint_dir_1"
    ckpt_path = create_checkpoint(ckpt_dir, 'a', 1.0)
    input_dict = {"ckpt_dir_or_file": ckpt_path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Valid checkpoint directory
    ckpt_dir = "checkpoint_dir_2"
    create_checkpoint(ckpt_dir, 'a', 1.0)
    input_dict = {"ckpt_dir_or_file": ckpt_dir}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Checkpoint with different name
    ckpt_dir = "checkpoint_dir_3"
    ckpt_path = create_checkpoint(ckpt_dir, 'c', 3.0) 
    input_dict = {"ckpt_dir_or_file": ckpt_path}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Nested directory for checkpoint
    ckpt_dir = os.path.join("checkpoint_dir_4", "nested")
    ckpt_path = create_checkpoint(ckpt_dir, 'a', 1.0)
    input_dict = {"ckpt_dir_or_file": ckpt_path}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Checkpoint path with long name
    long_name = "this_is_a_very_long_checkpoint_name_" * 5
    ckpt_dir = "checkpoint_dir_5"
    ckpt_path = create_checkpoint(ckpt_dir, long_name, 1.0)
    input_dict = {"ckpt_dir_or_file": ckpt_path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Directory contains the checkpoint path
    ckpt_dir = "checkpoint_dir_6"
    create_checkpoint(ckpt_dir, 'a', 1.0)
    input_dict = {"ckpt_dir_or_file": ckpt_dir}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Checkpoint path with dots in name
    ckpt_dir = "checkpoint_dir_7"
    ckpt_path = create_checkpoint(ckpt_dir, 'a.b.c', 1.0)
    input_dict = {"ckpt_dir_or_file": ckpt_path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Checkpoint path with dashes in name
    ckpt_dir = "checkpoint_dir_8"
    ckpt_path = create_checkpoint(ckpt_dir, 'a-b-c', 1.0)
    input_dict = {"ckpt_dir_or_file": ckpt_path}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Checkpoint path in a subdirectory with spaces
    ckpt_dir = os.path.join("checkpoint dir 9", "sub directory")
    ckpt_path = create_checkpoint(ckpt_dir, 'a', 1.0)
    input_dict = {"ckpt_dir_or_file": ckpt_path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Checkpoint with special characters in the path
    ckpt_dir = "checkpoint_dir_10" + "/@#$%^&*()"
    
    ckpt_dir = "checkpoint_dir_10/@#$%^&*()"
    ckpt_path = create_checkpoint(ckpt_dir, 'a', 1.0)
    input_dict = {"ckpt_dir_or_file": ckpt_path}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.train.load_checkpoint"] = tf_train_load_checkpoint_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.train.load_checkpoint' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.train.load_checkpoint'.")

check_valid('tf.train.load_checkpoint', generated_inputs['tf.train.load_checkpoint'], lib="tf", suffix=0)
