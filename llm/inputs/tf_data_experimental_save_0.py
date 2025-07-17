
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
import os

def tf_data_experimental_save_inputs():
    list_of_inputs = []

    # Input 1
    dataset = tf.data.Dataset.from_tensor_slices(np.array([1, 2, 3]))
    path = os.path.join(os.getcwd(), "saved_data_1")
    compression = "NONE"
    shard_func = tf.function(lambda x: tf.constant(0, dtype=tf.int64))
    checkpoint_args = []

    input_dict = {
        "dataset": dataset,
        "path": path,
        "compression": compression,
        "shard_func": shard_func,
        "checkpoint_args": checkpoint_args
    }
    list_of_inputs.append(input_dict)

    # Input 2
    dataset = tf.data.Dataset.from_tensor_slices(np.array([[1, 2], [3, 4]]))
    path = os.path.join(os.getcwd(), "saved_data_2")
    compression = "GZIP"
    shard_func = tf.function(lambda x: tf.constant(0, dtype=tf.int64))
    checkpoint_args = []

    input_dict = {
        "dataset": dataset,
        "path": path,
        "compression": compression,
        "shard_func": shard_func,
        "checkpoint_args": checkpoint_args
    }
    list_of_inputs.append(input_dict)

    # Input 3
    dataset = tf.data.Dataset.range(10)
    path = os.path.join(os.getcwd(), "saved_data_3")
    compression = "NONE"
    shard_func = tf.function(lambda x: tf.constant(0, dtype=tf.int64))
    checkpoint_args = []

    input_dict = {
        "dataset": dataset,
        "path": path,
        "compression": compression,
        "shard_func": shard_func,
        "checkpoint_args": checkpoint_args
    }
    list_of_inputs.append(input_dict)

    # Input 4
    dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0]))
    path = os.path.join(os.getcwd(), "saved_data_4")
    compression = "GZIP"
    shard_func = tf.function(lambda x: tf.constant(0, dtype=tf.int64))
    checkpoint_args = []

    input_dict = {
        "dataset": dataset,
        "path": path,
        "compression": compression,
        "shard_func": shard_func,
        "checkpoint_args": checkpoint_args
    }
    list_of_inputs.append(input_dict)

    # Input 5
    dataset = tf.data.Dataset.from_tensor_slices(np.array(["a", "b", "c"]))
    path = os.path.join(os.getcwd(), "saved_data_5")
    compression = "NONE"
    shard_func = tf.function(lambda x: tf.constant(0, dtype=tf.int64))
    checkpoint_args = []

    input_dict = {
        "dataset": dataset,
        "path": path,
        "compression": compression,
        "shard_func": shard_func,
        "checkpoint_args": checkpoint_args
    }
    list_of_inputs.append(input_dict)

    # Input 6
    dataset = tf.data.Dataset.from_tensor_slices(np.array([True, False, True]))
    path = os.path.join(os.getcwd(), "saved_data_6")
    compression = "GZIP"
    shard_func = tf.function(lambda x: tf.constant(0, dtype=tf.int64))
    checkpoint_args = []

    input_dict = {
        "dataset": dataset,
        "path": path,
        "compression": compression,
        "shard_func": shard_func,
        "checkpoint_args": checkpoint_args
    }
    list_of_inputs.append(input_dict)

    # Input 7
    dataset = tf.data.Dataset.from_tensor_slices(np.array([1, 2, 3, 4, 5]))
    path = os.path.join(os.getcwd(), "saved_data_7")
    compression = "NONE"
    shard_func = tf.function(lambda x: tf.constant(0, dtype=tf.int64))
    checkpoint_args = []

    input_dict = {
        "dataset": dataset,
        "path": path,
        "compression": compression,
        "shard_func": shard_func,
        "checkpoint_args": checkpoint_args
    }
    list_of_inputs.append(input_dict)

    # Input 8
    dataset = tf.data.Dataset.from_tensor_slices(np.array([[1, 2, 3], [4, 5, 6]]))
    path = os.path.join(os.getcwd(), "saved_data_8")
    compression = "GZIP"
    shard_func = tf.function(lambda x: tf.constant(0, dtype=tf.int64))
    checkpoint_args = []

    input_dict = {
        "dataset": dataset,
        "path": path,
        "compression": compression,
        "shard_func": shard_func,
        "checkpoint_args": checkpoint_args
    }
    list_of_inputs.append(input_dict)

    # Input 9
    dataset = tf.data.Dataset.range(100)
    path = os.path.join(os.getcwd(), "saved_data_9")
    compression = "NONE"
    shard_func = tf.function(lambda x: tf.constant(0, dtype=tf.int64))
    checkpoint_args = []

    input_dict = {
        "dataset": dataset,
        "path": path,
        "compression": compression,
        "shard_func": shard_func,
        "checkpoint_args": checkpoint_args
    }
    list_of_inputs.append(input_dict)

    # Input 10
    dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    path = os.path.join(os.getcwd(), "saved_data_10")
    compression = "GZIP"
    shard_func = tf.function(lambda x: tf.constant(0, dtype=tf.int64))
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
