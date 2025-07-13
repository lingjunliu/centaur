
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
import os

def tf_train_checkpoints_iterator_inputs():
    list_of_inputs = []

    # Helper function to create a dummy checkpoint directory and files
    def create_dummy_checkpoints(checkpoint_dir, num_checkpoints):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        for i in range(num_checkpoints):
            checkpoint_path = os.path.join(checkpoint_dir, f"model.ckpt-{i}")
            with open(checkpoint_path + ".index", "w") as f:
                f.write("dummy index")
            with open(checkpoint_path + ".data-00000-of-00001", "w") as f:
                f.write("dummy data")
            with open(checkpoint_path + ".meta", "w") as f:
                f.write("dummy meta")
        return checkpoint_dir


    # Input 1, valid
    checkpoint_dir = "dummy_checkpoint_dir_1"
    checkpoint_dir = create_dummy_checkpoints(checkpoint_dir, 3)
    min_interval_secs = 1
    timeout = 5
    timeout_fn = []

    input_dict = {
        "checkpoint_dir": checkpoint_dir,
        "min_interval_secs": min_interval_secs,
        "timeout": timeout,
        "timeout_fn": timeout_fn
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # cleanup for the next iteration to prevent errors
    tf.io.gfile.rmtree(checkpoint_dir)

    # Input 2, valid, with timeout_fn
    checkpoint_dir = "dummy_checkpoint_dir_2"
    checkpoint_dir = create_dummy_checkpoints(checkpoint_dir, 1)

    min_interval_secs = 0
    timeout = 2
    def my_timeout_fn():
        return True
    timeout_fn = [my_timeout_fn]  # Timeout function should be a list even if it contains a single function

    input_dict = {
        "checkpoint_dir": checkpoint_dir,
        "min_interval_secs": min_interval_secs,
        "timeout": timeout,
        "timeout_fn": timeout_fn
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    tf.io.gfile.rmtree(checkpoint_dir)

    # Input 3, valid, indefinite timeout
    checkpoint_dir = "dummy_checkpoint_dir_3"
    checkpoint_dir = create_dummy_checkpoints(checkpoint_dir, 2)
    min_interval_secs = 2
    timeout = None
    timeout_fn = []

    input_dict = {
        "checkpoint_dir": checkpoint_dir,
        "min_interval_secs": min_interval_secs,
        "timeout": timeout,
        "timeout_fn": timeout_fn
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    tf.io.gfile.rmtree(checkpoint_dir)

    # Input 4, valid, zero timeout
    checkpoint_dir = "dummy_checkpoint_dir_4"
    checkpoint_dir = create_dummy_checkpoints(checkpoint_dir, 1)
    min_interval_secs = 0
    timeout = 0
    timeout_fn = []

    input_dict = {
        "checkpoint_dir": checkpoint_dir,
        "min_interval_secs": min_interval_secs,
        "timeout": timeout,
        "timeout_fn": timeout_fn
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    tf.io.gfile.rmtree(checkpoint_dir)

    # Input 5, valid, longer min_interval_secs
    checkpoint_dir = "dummy_checkpoint_dir_5"
    checkpoint_dir = create_dummy_checkpoints(checkpoint_dir, 4)
    min_interval_secs = 5
    timeout = 10
    timeout_fn = []

    input_dict = {
        "checkpoint_dir": checkpoint_dir,
        "min_interval_secs": min_interval_secs,
        "timeout": timeout,
        "timeout_fn": timeout_fn
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    tf.io.gfile.rmtree(checkpoint_dir)

    # Input 6, valid, timeout_fn returns False once
    checkpoint_dir = "dummy_checkpoint_dir_6"
    checkpoint_dir = create_dummy_checkpoints(checkpoint_dir, 2)
    min_interval_secs = 0
    timeout = 3
    def my_timeout_fn_false_once():
        if not hasattr(my_timeout_fn_false_once, 'called'):
            my_timeout_fn_false_once.called = False
            return False
        else:
            return True
    timeout_fn = [my_timeout_fn_false_once]

    input_dict = {
        "checkpoint_dir": checkpoint_dir,
        "min_interval_secs": min_interval_secs,
        "timeout": timeout,
        "timeout_fn": timeout_fn
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    tf.io.gfile.rmtree(checkpoint_dir)

    # Input 7, valid, empty checkpoint dir
    checkpoint_dir = "dummy_checkpoint_dir_7"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    min_interval_secs = 1
    timeout = 5
    timeout_fn = []

    input_dict = {
        "checkpoint_dir": checkpoint_dir,
        "min_interval_secs": min_interval_secs,
        "timeout": timeout,
        "timeout_fn": timeout_fn
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    tf.io.gfile.rmtree(checkpoint_dir)

    # Input 8, valid, with subdirectories in checkpoint dir
    checkpoint_dir = "dummy_checkpoint_dir_8"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    subdir = os.path.join(checkpoint_dir, "subdir")
    os.makedirs(subdir)
    create_dummy_checkpoints(subdir, 1)

    min_interval_secs = 1
    timeout = 5
    timeout_fn = []

    input_dict = {
        "checkpoint_dir": checkpoint_dir,
        "min_interval_secs": min_interval_secs,
        "timeout": timeout,
        "timeout_fn": timeout_fn
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    tf.io.gfile.rmtree(checkpoint_dir)

    # Input 9, valid, longer timeout and min_interval_secs
    checkpoint_dir = "dummy_checkpoint_dir_9"
    checkpoint_dir = create_dummy_checkpoints(checkpoint_dir, 1)
    min_interval_secs = 60
    timeout = 120
    timeout_fn = []

    input_dict = {
        "checkpoint_dir": checkpoint_dir,
        "min_interval_secs": min_interval_secs,
        "timeout": timeout,
        "timeout_fn": timeout_fn
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    tf.io.gfile.rmtree(checkpoint_dir)

    # Input 10, valid, timeout_fn does nothing
    checkpoint_dir = "dummy_checkpoint_dir_10"
    checkpoint_dir = create_dummy_checkpoints(checkpoint_dir, 1)
    min_interval_secs = 0
    timeout = 2
    def my_timeout_fn_noop():
        return False
    timeout_fn = [my_timeout_fn_noop]

    input_dict = {
        "checkpoint_dir": checkpoint_dir,
        "min_interval_secs": min_interval_secs,
        "timeout": timeout,
        "timeout_fn": timeout_fn
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    tf.io.gfile.rmtree(checkpoint_dir)

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.train.checkpoints_iterator"] = tf_train_checkpoints_iterator_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.train.checkpoints_iterator' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.train.checkpoints_iterator'.")

check_valid('tf.train.checkpoints_iterator', generated_inputs['tf.train.checkpoints_iterator'], lib="tf", suffix=0)
