
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_train_checkpointmanager_inputs():
    list_of_inputs = []

    # Input 1
    checkpoint = tf.train.Checkpoint(v=tf.Variable(1.0))
    directory = "/tmp/model_1"
    max_to_keep = np.int32(5)
    keep_checkpoint_every_n_hours = np.float32(0.5)
    checkpoint_name = "ckpt"
    step_counter = np.int32(100)
    checkpoint_interval = np.int32(1000)
    init_fn = tf.constant(1.0)

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
    checkpoint = tf.train.Checkpoint(v=tf.Variable(1.0))
    directory = "/tmp/model_2"
    max_to_keep = np.int32(1)
    keep_checkpoint_every_n_hours = np.float32(1.0)
    checkpoint_name = "model"
    step_counter = np.int32(0)
    checkpoint_interval = np.int32(500)
    init_fn = tf.constant([1, 2, 3])

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
    checkpoint = tf.train.Checkpoint(v=tf.Variable(1.0))
    directory = "/tmp/model_3"
    max_to_keep = np.int32(10)
    keep_checkpoint_every_n_hours = np.float32(2.5)
    checkpoint_name = "backup"
    step_counter = np.int32(1000)
    checkpoint_interval = np.int32(2000)
    init_fn = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)

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
    checkpoint = tf.train.Checkpoint(v=tf.Variable(1.0))
    directory = "/tmp/model_4"
    max_to_keep = np.int32(3)
    keep_checkpoint_every_n_hours = np.float32(0.1)
    checkpoint_name = "final"
    step_counter = np.int32(50)
    checkpoint_interval = np.int32(100)
    init_fn = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=tf.float32)

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
    checkpoint = tf.train.Checkpoint(v=tf.Variable(1.0))
    directory = "/tmp/model_5"
    max_to_keep = np.int32(7)
    keep_checkpoint_every_n_hours = np.float32(12.0)
    checkpoint_name = "epoch"
    step_counter = np.int32(2500)
    checkpoint_interval = np.int32(5000)
    init_fn = tf.constant(5, dtype=tf.float32)

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
    checkpoint = tf.train.Checkpoint(v=tf.Variable(1.0))
    directory = "/tmp/model_6"
    max_to_keep = np.int32(2)
    keep_checkpoint_every_n_hours = np.float32(0.01)
    checkpoint_name = "initial"
    step_counter = np.int32(1)
    checkpoint_interval = np.int32(2)
    init_fn = tf.constant(10.0)

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
    checkpoint = tf.train.Checkpoint(v=tf.Variable(1.0))
    directory = "gs://bucket/model_7"
    max_to_keep = np.int32(4)
    keep_checkpoint_every_n_hours = np.float32(6.0)
    checkpoint_name = "stage"
    step_counter = np.int32(750)
    checkpoint_interval = np.int32(1500)
    init_fn = tf.constant([1, 2, 3, 4, 5], dtype=tf.float32)

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
    checkpoint = tf.train.Checkpoint(v=tf.Variable(1.0))
    directory = "/another/path/model_8"
    max_to_keep = np.int32(15)
    keep_checkpoint_every_n_hours = np.float32(24.0)
    checkpoint_name = "long_run"
    step_counter = np.int32(50000)
    checkpoint_interval = np.int32(100000)
    init_fn = tf.constant(np.arange(24).reshape((2,3,4)), dtype=tf.float32)

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
    checkpoint = tf.train.Checkpoint(v=tf.Variable(1.0))
    directory = "/tmp/model_9"
    max_to_keep = np.int32(1)
    keep_checkpoint_every_n_hours = np.float32(0.5)
    checkpoint_name = "test_model"
    step_counter = np.int32(10)
    checkpoint_interval = np.int32(100)
    init_fn = tf.constant(5.0)

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
    checkpoint = tf.train.Checkpoint(v=tf.Variable(1.0))
    directory = "/tmp/model_10"
    max_to_keep = np.int32(20)
    keep_checkpoint_every_n_hours = np.float32(48.0)
    checkpoint_name = "very_long_run"
    step_counter = np.int32(1000000)
    checkpoint_interval = np.int32(2000000)
    init_fn = tf.constant(100, dtype=tf.float32)

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

generated_inputs["tf.train.CheckpointManager"] = tf_train_checkpointmanager_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.train.CheckpointManager' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.train.CheckpointManager'.")

check_valid('tf.train.CheckpointManager', generated_inputs['tf.train.CheckpointManager'], lib="tf", suffix=0)
