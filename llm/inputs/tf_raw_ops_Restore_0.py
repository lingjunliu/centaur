
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_restore_inputs():
    list_of_inputs = []

    # Input 1
    file_pattern = tf.constant("checkpoint_file_1", dtype=tf.string)
    tensor_name = tf.constant("tensor_a", dtype=tf.string)
    dt = np.float32
    preferred_shard = -1
    name = "restore_op_1"

    input_dict = {
        "file_pattern": file_pattern,
        "tensor_name": tensor_name,
        "dt": dt,
        "preferred_shard": preferred_shard,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    file_pattern = tf.constant("checkpoint_file_2*", dtype=tf.string)
    tensor_name = tf.constant("tensor_b", dtype=tf.string)
    dt = np.int32
    preferred_shard = 0
    name = "restore_op_2"

    input_dict = {
        "file_pattern": file_pattern,
        "tensor_name": tensor_name,
        "dt": dt,
        "preferred_shard": preferred_shard,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    file_pattern = tf.constant("checkpoint_file_3?", dtype=tf.string)
    tensor_name = tf.constant("tensor_c", dtype=tf.string)
    dt = np.int64
    preferred_shard = 1
    name = "restore_op_3"

    input_dict = {
        "file_pattern": file_pattern,
        "tensor_name": tensor_name,
        "dt": dt,
        "preferred_shard": preferred_shard,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    file_pattern = tf.constant("checkpoint_file_4", dtype=tf.string)
    tensor_name = tf.constant("tensor_d", dtype=tf.string)
    dt = np.float64
    preferred_shard = 2
    name = "restore_op_4"

    input_dict = {
        "file_pattern": file_pattern,
        "tensor_name": tensor_name,
        "dt": dt,
        "preferred_shard": preferred_shard,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    file_pattern = tf.constant("checkpoint_file_5", dtype=tf.string)
    tensor_name = tf.constant("tensor_e", dtype=tf.string)
    dt = np.complex64
    preferred_shard = -1
    name = "restore_op_5"

    input_dict = {
        "file_pattern": file_pattern,
        "tensor_name": tensor_name,
        "dt": dt,
        "preferred_shard": preferred_shard,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6
    file_pattern = tf.constant("checkpoint_file_6*", dtype=tf.string)
    tensor_name = tf.constant("tensor_f", dtype=tf.string)
    dt = np.complex128
    preferred_shard = 0
    name = "restore_op_6"

    input_dict = {
        "file_pattern": file_pattern,
        "tensor_name": tensor_name,
        "dt": dt,
        "preferred_shard": preferred_shard,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    file_pattern = tf.constant("checkpoint_file_7?", dtype=tf.string)
    tensor_name = tf.constant("tensor_g", dtype=tf.string)
    dt = np.bfloat16
    preferred_shard = 1
    name = "restore_op_7"

    input_dict = {
        "file_pattern": file_pattern,
        "tensor_name": tensor_name,
        "dt": dt,
        "preferred_shard": preferred_shard,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    file_pattern = tf.constant("checkpoint_file_8", dtype=tf.string)
    tensor_name = tf.constant("tensor_h", dtype=tf.string)
    dt = np.uint8
    preferred_shard = 2
    name = "restore_op_8"

    input_dict = {
        "file_pattern": file_pattern,
        "tensor_name": tensor_name,
        "dt": dt,
        "preferred_shard": preferred_shard,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    file_pattern = tf.constant("checkpoint_file_9", dtype=tf.string)
    tensor_name = tf.constant("tensor_i", dtype=tf.string)
    dt = np.bool_
    preferred_shard = -1
    name = "restore_op_9"

    input_dict = {
        "file_pattern": file_pattern,
        "tensor_name": tensor_name,
        "dt": dt,
        "preferred_shard": preferred_shard,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    file_pattern = tf.constant("checkpoint_file_10*", dtype=tf.string)
    tensor_name = tf.constant("tensor_j", dtype=tf.string)
    dt = np.int8
    preferred_shard = 0
    name = "restore_op_10"

    input_dict = {
        "file_pattern": file_pattern,
        "tensor_name": tensor_name,
        "dt": dt,
        "preferred_shard": preferred_shard,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Restore"] = tf_raw_ops_restore_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Restore' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Restore'.")

check_valid('tf.raw_ops.Restore', generated_inputs['tf.raw_ops.Restore'], lib="tf", suffix=0)
