
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
    dataset = np.array([1, 2, 3], dtype=np.int64)
    path = os.path.join(os.getcwd(), "saved_data_1")
    compression = None
    shard_func = None
    checkpoint_args = []
    input_dict = {"dataset": dataset, "path": path, "compression": compression, "shard_func": shard_func, "checkpoint_args": checkpoint_args}
    list_of_inputs.append(input_dict)

    # Input 2: With compression
    dataset = np.array([4, 5, 6], dtype=np.int64)
    path = os.path.join(os.getcwd(), "saved_data_2")
    compression = "GZIP"
    shard_func = None
    checkpoint_args = []
    input_dict = {"dataset": dataset, "path": path, "compression": compression, "shard_func": shard_func, "checkpoint_args": checkpoint_args}
    list_of_inputs.append(input_dict)

    # Input 3: With shard_func
    dataset = np.array([7, 8, 9, 10], dtype=np.int64)

    def shard_func(element):
        return np.int64(element % 2)

    path = os.path.join(os.getcwd(), "saved_data_3")
    compression = None
    checkpoint_args = []

    input_dict = {"dataset": dataset, "path": path, "compression": compression, "shard_func": shard_func, "checkpoint_args": checkpoint_args}
    list_of_inputs.append(input_dict)

    # Input 4: With checkpoint_args (empty)
    dataset = np.array([11, 12, 13], dtype=np.int64)
    path = os.path.join(os.getcwd(), "saved_data_4")
    compression = None
    shard_func = None
    checkpoint_args = []
    input_dict = {"dataset": dataset, "path": path, "compression": compression, "shard_func": shard_func, "checkpoint_args": checkpoint_args}
    list_of_inputs.append(input_dict)

    # Input 5: Different dataset type (string)
    dataset = np.array(["a", "b", "c"])
    path = os.path.join(os.getcwd(), "saved_data_5")
    compression = None
    shard_func = None
    checkpoint_args = []
    input_dict = {"dataset": dataset, "path": path, "compression": compression, "shard_func": shard_func, "checkpoint_args": checkpoint_args}
    list_of_inputs.append(input_dict)

    # Input 6: Multi-dimensional dataset
    dataset = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64)
    path = os.path.join(os.getcwd(), "saved_data_6")
    compression = None
    shard_func = None
    checkpoint_args = []
    input_dict = {"dataset": dataset, "path": path, "compression": compression, "shard_func": shard_func, "checkpoint_args": checkpoint_args}
    list_of_inputs.append(input_dict)

    # Input 7: More complex shard_func
    dataset = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)

    def shard_func_complex(element):
        return np.int64((element * 2 + 1) % 3)

    path = os.path.join(os.getcwd(), "saved_data_7")
    compression = None
    checkpoint_args = []
    input_dict = {"dataset": dataset, "path": path, "compression": compression, "shard_func": shard_func_complex, "checkpoint_args": checkpoint_args}
    list_of_inputs.append(input_dict)

     # Input 8: Dataset with floats
    dataset = np.array([1.1, 2.2, 3.3], dtype=np.float64)
    path = os.path.join(os.getcwd(), "saved_data_8")
    compression = None
    shard_func = None
    checkpoint_args = []
    input_dict = {"dataset": dataset, "path": path, "compression": compression, "shard_func": shard_func, "checkpoint_args": checkpoint_args}
    list_of_inputs.append(input_dict)

    # Input 9: Dataset with booleans
    dataset = np.array([True, False, True])
    path = os.path.join(os.getcwd(), "saved_data_9")
    compression = None
    shard_func = None
    checkpoint_args = []
    input_dict = {"dataset": dataset, "path": path, "compression": compression, "shard_func": shard_func, "checkpoint_args": checkpoint_args}
    list_of_inputs.append(input_dict)

    # Input 10: empty dataset
    dataset = np.array([], dtype=np.int64)
    path = os.path.join(os.getcwd(), "saved_data_10")
    compression = None
    shard_func = None
    checkpoint_args = []
    input_dict = {"dataset": dataset, "path": path, "compression": compression, "shard_func": shard_func, "checkpoint_args": checkpoint_args}
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
