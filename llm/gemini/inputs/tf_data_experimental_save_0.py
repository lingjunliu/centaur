
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
    dataset = np.array([1, 2, 3, 4, 5])
    path = os.path.join(os.getcwd(), "saved_dataset_1")
    compression = "NONE"
    shard_func = lambda x: np.int64(x % 2)

    input_dict = {
        "dataset": dataset,
        "path": path,
        "compression": compression,
        "shard_func": shard_func,
        "checkpoint_args": []
    }
    list_of_inputs.append(input_dict)

    # Input 2
    dataset = np.array([[1, 2], [3, 4], [5, 6]])
    path = os.path.join(os.getcwd(), "saved_dataset_2")
    compression = "GZIP"
    shard_func = lambda x: np.int64(0)

    input_dict = {
        "dataset": dataset,
        "path": path,
        "compression": compression,
        "shard_func": shard_func,
        "checkpoint_args": []
    }
    list_of_inputs.append(input_dict)

    # Input 3
    dataset = np.array(range(10))
    path = os.path.join(os.getcwd(), "saved_dataset_3")
    compression = "NONE"
    shard_func = lambda x: np.int64(x // 3)

    input_dict = {
        "dataset": dataset,
        "path": path,
        "compression": compression,
        "shard_func": shard_func,
        "checkpoint_args": []
    }
    list_of_inputs.append(input_dict)

   # Input 4
    dataset = np.array([[-1, -2], [-3, -4]])
    path = os.path.join(os.getcwd(), "saved_dataset_4")
    compression = "GZIP"
    shard_func = lambda x: np.int64(1)

    input_dict = {
        "dataset": dataset,
        "path": path,
        "compression": compression,
        "shard_func": shard_func,
        "checkpoint_args": []
    }
    list_of_inputs.append(input_dict)

    # Input 5
    dataset = np.array([1.0, 2.0, 3.0])
    path = os.path.join(os.getcwd(), "saved_dataset_5")
    compression = "NONE"
    shard_func = lambda x: np.int64(x)

    input_dict = {
        "dataset": dataset,
        "path": path,
        "compression": compression,
        "shard_func": shard_func,
        "checkpoint_args": []
    }
    list_of_inputs.append(input_dict)

    # Input 6
    dataset = np.array(["a", "b", "c"])
    path = os.path.join(os.getcwd(), "saved_dataset_6")
    compression = "GZIP"
    shard_func = lambda x: np.int64(0 if x == "a" else 1)

    input_dict = {
        "dataset": dataset,
        "path": path,
        "compression": compression,
        "shard_func": shard_func,
        "checkpoint_args": []
    }
    list_of_inputs.append(input_dict)

    # Input 8: Empty dataset
    dataset = np.array([])
    path = os.path.join(os.getcwd(), "saved_dataset_8")
    compression = "GZIP"
    shard_func = lambda x: np.int64(0) if len(x)>0 else np.int64(0)

    input_dict = {
        "dataset": dataset,
        "path": path,
        "compression": compression,
        "shard_func": shard_func,
        "checkpoint_args": []
    }
    list_of_inputs.append(input_dict)

   # Input 9: Larger Dataset with complex sharding
    dataset = np.array(range(100))
    path = os.path.join(os.getcwd(), "saved_dataset_9")
    compression = "NONE"
    shard_func = lambda x: np.int64(x % 10)

    input_dict = {
        "dataset": dataset,
        "path": path,
        "compression": compression,
        "shard_func": shard_func,
        "checkpoint_args": []
    }
    list_of_inputs.append(input_dict)

    # Input 10: Strings
    dataset = np.array(["test1", "test2", "test3"])
    path = os.path.join(os.getcwd(), "saved_dataset_10")
    compression = "NONE"
    shard_func = lambda x: np.int64(len(x) % 2)

    input_dict = {
        "dataset": dataset,
        "path": path,
        "compression": compression,
        "shard_func": shard_func,
        "checkpoint_args": []
    }
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.save"] = tf_data_experimental_save_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.save' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.save'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.data.experimental.save', generated_inputs['tf.data.experimental.save'], lib="tf", suffix=0)
