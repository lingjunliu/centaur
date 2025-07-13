
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
import os

def tf_data_experimental_save_inputs():
    list_of_inputs = []

    # Input 1: Minimal valid case
    dataset = tf.constant(np.array([0, 1, 2, 3, 4]))
    path = os.path.join(os.getcwd(), "test_data_1")
    compression = "NONE"
    shard_func = None
    checkpoint_args = []

    input_dict = {
        "dataset": dataset,
        "path": path,
        "compression": compression,
        "shard_func": shard_func,
        "checkpoint_args": checkpoint_args
    }
    list_of_inputs.append(input_dict)

    # Input 2: With GZIP compression
    dataset = tf.constant(np.array([1, 2, 3, 4, 5]))
    path = os.path.join(os.getcwd(), "test_data_2")
    compression = "GZIP"
    shard_func = None
    checkpoint_args = []

    input_dict = {
        "dataset": dataset,
        "path": path,
        "compression": compression,
        "shard_func": shard_func,
        "checkpoint_args": checkpoint_args
    }
    list_of_inputs.append(input_dict)

    # Input 3: With checkpoint args
    dataset = tf.constant(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    path = os.path.join(os.getcwd(), "test_data_4")
    compression = "NONE"
    shard_func = None
    step_counter = tf.Variable(0, trainable=False)
    checkpoint_args = [{
        "checkpoint_interval": 2,
        "step_counter": step_counter.numpy(), #convert step_counter to numpy to resolve the error
        "directory": path,
        "max_to_keep": 5
    }]

    input_dict = {
        "dataset": dataset,
        "path": path,
        "compression": compression,
        "shard_func": shard_func,
        "checkpoint_args": checkpoint_args
    }
    list_of_inputs.append(input_dict)

     # Input 4: Different dataset type (string)
    dataset = tf.constant(np.array(["a", "b", "c"]))
    path = os.path.join(os.getcwd(), "test_data_5")
    compression = "NONE"
    shard_func = None
    checkpoint_args = []

    input_dict = {
        "dataset": dataset,
        "path": path,
        "compression": compression,
        "shard_func": shard_func,
        "checkpoint_args": checkpoint_args
    }
    list_of_inputs.append(input_dict)

    # Input 5: Larger dataset and path
    dataset = tf.constant(np.arange(1000))
    path = os.path.join(os.getcwd(), "very_long_path_to_store_large_dataset")
    compression = "GZIP"
    shard_func = None
    checkpoint_args = []

    input_dict = {
        "dataset": dataset,
        "path": path,
        "compression": compression,
        "shard_func": shard_func,
        "checkpoint_args": checkpoint_args
    }
    list_of_inputs.append(input_dict)

    # Input 6: Multidimensional dataset
    dataset = tf.constant(np.array([[1, 2], [3, 4], [5, 6]]))
    path = os.path.join(os.getcwd(), "test_data_8")
    compression = "NONE"
    shard_func = None
    checkpoint_args = []

    input_dict = {
        "dataset": dataset,
        "path": path,
        "compression": compression,
        "shard_func": shard_func,
        "checkpoint_args": checkpoint_args
    }
    list_of_inputs.append(input_dict)

    # Input 7: Empty dataset
    dataset = tf.constant(np.array([]))
    path = os.path.join(os.getcwd(), "test_data_9")
    compression = "NONE"
    shard_func = None
    checkpoint_args = []

    input_dict = {
        "dataset": dataset,
        "path": path,
        "compression": compression,
        "shard_func": shard_func,
        "checkpoint_args": checkpoint_args
    }
    list_of_inputs.append(input_dict)

    # Input 8: checkpoint_args with different parameters
    dataset = tf.constant(np.arange(50))
    path = os.path.join(os.getcwd(), "test_data_10")
    compression = "NONE"
    shard_func = None
    step_counter = tf.Variable(0, trainable=False)
    checkpoint_args = [{
        "checkpoint_interval": 10,
        "step_counter": step_counter.numpy(),#convert step_counter to numpy to resolve the error
        "directory": path,
        "max_to_keep": 10
    }]

    input_dict = {
        "dataset": dataset,
        "path": path,
        "compression": compression,
        "shard_func": shard_func,
        "checkpoint_args": checkpoint_args
    }
    list_of_inputs.append(input_dict)

     # Input 9: Different dataset type (int64)
    dataset = tf.constant(np.array([100000000000, 200000000000, 300000000000], dtype=np.int64))
    path = os.path.join(os.getcwd(), "test_data_11")
    compression = "NONE"
    shard_func = None
    checkpoint_args = []

    input_dict = {
        "dataset": dataset,
        "path": path,
        "compression": compression,
        "shard_func": shard_func,
        "checkpoint_args": checkpoint_args
    }
    list_of_inputs.append(input_dict)

      # Input 10: Different dataset type (float32)
    dataset = tf.constant(np.array([1.1, 2.2, 3.3], dtype=np.float32))
    path = os.path.join(os.getcwd(), "test_data_12")
    compression = "NONE"
    shard_func = None
    checkpoint_args = []

    input_dict = {
        "dataset": dataset,
        "path": path,
        "compression": compression,
        "shard_func": shard_func,
        "checkpoint_args": checkpoint_args
    }
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.save"] = tf_data_experimental_save_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.data.experimental.save' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.save'.")

check_valid('tf.data.experimental.save', generated_inputs['tf.data.experimental.save'], lib="tf", suffix=0)
