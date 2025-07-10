
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
import os

def tf_data_experimental_save_inputs():
    list_of_inputs = []

    # Input 1: Minimal example
    dataset = tf.constant(np.array([1, 2, 3]))
    path = os.path.join("./tmp", "saved_data1")
    compression = "NONE"
    shard_func = None
    checkpoint_args = None
    input_dict = {"dataset": dataset, "path": path, "compression": compression, "shard_func": shard_func, "checkpoint_args": checkpoint_args}
    list_of_inputs.append(input_dict)

    # Input 2: With GZIP compression
    dataset = tf.constant(np.array([4, 5, 6]))
    path = os.path.join("./tmp", "saved_data2")
    compression = "GZIP"
    shard_func = None
    checkpoint_args = None
    input_dict = {"dataset": dataset, "path": path, "compression": compression, "shard_func": shard_func, "checkpoint_args": checkpoint_args}
    list_of_inputs.append(input_dict)

    # Input 3: With shard_func
    dataset = tf.constant(np.array([7, 8, 9, 10]))
    path = os.path.join("./tmp", "saved_data3")
    compression = "NONE"
    def shard_func_example(element):
        return tf.constant(np.int64(element % 2))
    shard_func = shard_func_example
    checkpoint_args = None

    input_dict = {"dataset": dataset, "path": path, "compression": compression, "shard_func": shard_func, "checkpoint_args": checkpoint_args}
    list_of_inputs.append(input_dict)

    # Input 4: With checkpoint_args
    dataset = tf.constant(np.array([11, 12, 13, 14]))
    path = os.path.join("./tmp", "saved_data4")
    compression = "NONE"
    shard_func = None
    step_counter = tf.constant(np.array(0))
    checkpoint_args = [{
        "checkpoint_interval": tf.constant(np.array(2)),
        "step_counter": step_counter,
        "directory": os.path.join("./tmp", "checkpoints"),
        "max_to_keep": tf.constant(np.array(5)),
    }]
    input_dict = {"dataset": dataset, "path": path, "compression": compression, "shard_func": shard_func, "checkpoint_args": checkpoint_args}
    list_of_inputs.append(input_dict)

    # Input 5: Larger dataset, shard_func, and GZIP
    dataset = tf.constant(np.arange(100))
    path = os.path.join("./tmp", "saved_data5")
    compression = "GZIP"
    def shard_func_example2(element):
        return tf.constant(np.int64(element % 10))
    shard_func = shard_func_example2
    checkpoint_args = None
    input_dict = {"dataset": dataset, "path": path, "compression": compression, "shard_func": shard_func, "checkpoint_args": checkpoint_args}
    list_of_inputs.append(input_dict)

    # Input 6: Multi-dimensional dataset
    dataset = tf.constant(np.array([[1, 2], [3, 4], [5, 6]]))
    path = os.path.join("./tmp", "saved_data6")
    compression = "NONE"
    shard_func = None
    checkpoint_args = None
    input_dict = {"dataset": dataset, "path": path, "compression": compression, "shard_func": shard_func, "checkpoint_args": checkpoint_args}
    list_of_inputs.append(input_dict)

    # Input 7: Dataset with strings
    dataset = tf.constant(np.array([b"a", b"b", b"c"]))
    path = os.path.join("./tmp", "saved_data7")
    compression = "NONE"
    shard_func = None
    checkpoint_args = None
    input_dict = {"dataset": dataset, "path": path, "compression": compression, "shard_func": shard_func, "checkpoint_args": checkpoint_args}
    list_of_inputs.append(input_dict)

    # Input 8: Dataset with different data types
    dataset = tf.constant(np.array([1, 2, 3]))
    path = os.path.join("./tmp", "saved_data8")
    compression = "NONE"
    shard_func = None
    checkpoint_args = None
    input_dict = {"dataset": dataset, "path": path, "compression": compression, "shard_func": shard_func, "checkpoint_args": checkpoint_args}
    list_of_inputs.append(input_dict)
    
    # Input 9: Larger checkpoint interval
    dataset = tf.constant(np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20]))
    path = os.path.join("./tmp", "saved_data9")
    compression = "NONE"
    shard_func = None
    step_counter = tf.constant(np.array(0))
    checkpoint_args = [{
        "checkpoint_interval": tf.constant(np.array(5)),
        "step_counter": step_counter,
        "directory": os.path.join("./tmp", "checkpoints2"),
        "max_to_keep": tf.constant(np.array(3)),
    }]
    input_dict = {"dataset": dataset, "path": path, "compression": compression, "shard_func": shard_func, "checkpoint_args": checkpoint_args}
    list_of_inputs.append(input_dict)
    
    # Input 10: Using a different shard_func logic
    dataset = tf.constant(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    path = os.path.join("./tmp", "saved_data10")
    compression = "NONE"
    def shard_func_example3(element):
        return tf.constant(np.int64((element * 2) % 5))
    shard_func = shard_func_example3
    checkpoint_args = None

    input_dict = {"dataset": dataset, "path": path, "compression": compression, "shard_func": shard_func, "checkpoint_args": checkpoint_args}
    list_of_inputs.append(input_dict)

    # Input 11: Empty dataset
    dataset = tf.constant(np.array([]))
    path = os.path.join("./tmp", "saved_data11")
    compression = "NONE"
    shard_func = None
    checkpoint_args = None
    input_dict = {"dataset": dataset, "path": path, "compression": compression, "shard_func": shard_func, "checkpoint_args": checkpoint_args}
    list_of_inputs.append(input_dict)

    # Input 12: checkpoint_args as an empty list
    dataset = tf.constant(np.array([1, 2, 3]))
    path = os.path.join("./tmp", "saved_data12")
    compression = "NONE"
    shard_func = None
    checkpoint_args = []
    input_dict = {"dataset": dataset, "path": path, "compression": compression, "shard_func": shard_func, "checkpoint_args": checkpoint_args}
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs["tf.data.experimental.save"] = tf_data_experimental_save_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.data.experimental.save' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.save'.")

check_valid('tf.data.experimental.save', generated_inputs['tf.data.experimental.save'], lib="tf", suffix=0)
