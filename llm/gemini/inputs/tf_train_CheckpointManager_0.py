
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_train_CheckpointManager_inputs():
    list_of_inputs = []

    # Input 1
    checkpoint = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    directory = "/tmp/model1"
    max_to_keep = 5
    keep_checkpoint_every_n_hours = 0.5
    checkpoint_name = "ckpt1"
    step_counter = 10
    checkpoint_interval = 100
    init_fn = np.array(1.0, dtype=np.float32)

    input_dict = {
        "checkpoint": checkpoint,
        "directory": directory,
        "max_to_keep": max_to_keep,
        "keep_checkpoint_every_n_hours": keep_checkpoint_every_n_hours,
        "checkpoint_name": checkpoint_name,
        "step_counter": step_counter,
        "checkpoint_interval": checkpoint_interval,
        "init_fn": init_fn
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    checkpoint = np.array([[1, 2], [3, 4]], dtype=np.int32)
    directory = "/tmp/model2"
    max_to_keep = 10
    keep_checkpoint_every_n_hours = 1.0
    checkpoint_name = "ckpt2"
    step_counter = 20
    checkpoint_interval = 200
    init_fn = np.array(2.0, dtype=np.float32)

    input_dict = {
        "checkpoint": checkpoint,
        "directory": directory,
        "max_to_keep": max_to_keep,
        "keep_checkpoint_every_n_hours": keep_checkpoint_every_n_hours,
        "checkpoint_name": checkpoint_name,
        "step_counter": step_counter,
        "checkpoint_interval": checkpoint_interval,
        "init_fn": init_fn
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    checkpoint = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float64)
    directory = "/tmp/model3"
    max_to_keep = 3
    keep_checkpoint_every_n_hours = 2.0
    checkpoint_name = "ckpt3"
    step_counter = 5
    checkpoint_interval = 50
    init_fn = np.array(3.0, dtype=np.float32)

    input_dict = {
        "checkpoint": checkpoint,
        "directory": directory,
        "max_to_keep": max_to_keep,
        "keep_checkpoint_every_n_hours": keep_checkpoint_every_n_hours,
        "checkpoint_name": checkpoint_name,
        "step_counter": step_counter,
        "checkpoint_interval": checkpoint_interval,
        "init_fn": init_fn
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    checkpoint = np.array([-1, -2, -3], dtype=np.int64)
    directory = "/tmp/model4"
    max_to_keep = 7
    keep_checkpoint_every_n_hours = 0.25
    checkpoint_name = "ckpt4"
    step_counter = 15
    checkpoint_interval = 150
    init_fn = np.array(4.0, dtype=np.float32)

    input_dict = {
        "checkpoint": checkpoint,
        "directory": directory,
        "max_to_keep": max_to_keep,
        "keep_checkpoint_every_n_hours": keep_checkpoint_every_n_hours,
        "checkpoint_name": checkpoint_name,
        "step_counter": step_counter,
        "checkpoint_interval": checkpoint_interval,
        "init_fn": init_fn
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    checkpoint = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    directory = "/tmp/model5"
    max_to_keep = 1
    keep_checkpoint_every_n_hours = 0.1
    checkpoint_name = "ckpt5"
    step_counter = 1
    checkpoint_interval = 10
    init_fn = np.array(5.0, dtype=np.float32)

    input_dict = {
        "checkpoint": checkpoint,
        "directory": directory,
        "max_to_keep": max_to_keep,
        "keep_checkpoint_every_n_hours": keep_checkpoint_every_n_hours,
        "checkpoint_name": checkpoint_name,
        "step_counter": step_counter,
        "checkpoint_interval": checkpoint_interval,
        "init_fn": init_fn
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    checkpoint = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    directory = "/tmp/model6"
    max_to_keep = 4
    keep_checkpoint_every_n_hours = 3.0
    checkpoint_name = "ckpt6"
    step_counter = 40
    checkpoint_interval = 400
    init_fn = np.array(6.0, dtype=np.float32)

    input_dict = {
        "checkpoint": checkpoint,
        "directory": directory,
        "max_to_keep": max_to_keep,
        "keep_checkpoint_every_n_hours": keep_checkpoint_every_n_hours,
        "checkpoint_name": checkpoint_name,
        "step_counter": step_counter,
        "checkpoint_interval": checkpoint_interval,
        "init_fn": init_fn
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    checkpoint = np.array([1.1, 2.2, 3.3, 4.4], dtype=np.float64)
    directory = "/tmp/model7"
    max_to_keep = 6
    keep_checkpoint_every_n_hours = 1.5
    checkpoint_name = "ckpt7"
    step_counter = 7
    checkpoint_interval = 70
    init_fn = np.array(7.0, dtype=np.float32)

    input_dict = {
        "checkpoint": checkpoint,
        "directory": directory,
        "max_to_keep": max_to_keep,
        "keep_checkpoint_every_n_hours": keep_checkpoint_every_n_hours,
        "checkpoint_name": checkpoint_name,
        "step_counter": step_counter,
        "checkpoint_interval": checkpoint_interval,
        "init_fn": init_fn
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8
    checkpoint = np.array([-10, -20], dtype=np.int64)
    directory = "/tmp/model8"
    max_to_keep = 2
    keep_checkpoint_every_n_hours = 0.75
    checkpoint_name = "ckpt8"
    step_counter = 25
    checkpoint_interval = 250
    init_fn = np.array(8.0, dtype=np.float32)

    input_dict = {
        "checkpoint": checkpoint,
        "directory": directory,
        "max_to_keep": max_to_keep,
        "keep_checkpoint_every_n_hours": keep_checkpoint_every_n_hours,
        "checkpoint_name": checkpoint_name,
        "step_counter": step_counter,
        "checkpoint_interval": checkpoint_interval,
        "init_fn": init_fn
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    checkpoint = np.array([0.4, 0.5, 0.6], dtype=np.float32)
    directory = "/tmp/model9"
    max_to_keep = 9
    keep_checkpoint_every_n_hours = 0.05
    checkpoint_name = "ckpt9"
    step_counter = 3
    checkpoint_interval = 30
    init_fn = np.array(9.0, dtype=np.float32)

    input_dict = {
        "checkpoint": checkpoint,
        "directory": directory,
        "max_to_keep": max_to_keep,
        "keep_checkpoint_every_n_hours": keep_checkpoint_every_n_hours,
        "checkpoint_name": checkpoint_name,
        "step_counter": step_counter,
        "checkpoint_interval": checkpoint_interval,
        "init_fn": init_fn
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    checkpoint = np.array([[7, 8], [9, 10]], dtype=np.int32)
    directory = "/tmp/model10"
    max_to_keep = 8
    keep_checkpoint_every_n_hours = 4.0
    checkpoint_name = "ckpt10"
    step_counter = 80
    checkpoint_interval = 800
    init_fn = np.array(10.0, dtype=np.float32)

    input_dict = {
        "checkpoint": checkpoint,
        "directory": directory,
        "max_to_keep": max_to_keep,
        "keep_checkpoint_every_n_hours": keep_checkpoint_every_n_hours,
        "checkpoint_name": checkpoint_name,
        "step_counter": step_counter,
        "checkpoint_interval": checkpoint_interval,
        "init_fn": init_fn
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.train.CheckpointManager"] = tf_train_CheckpointManager_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.train.CheckpointManager' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.train.CheckpointManager'.")

check_valid('tf.train.CheckpointManager', generated_inputs['tf.train.CheckpointManager'], lib="tf", suffix=0)
