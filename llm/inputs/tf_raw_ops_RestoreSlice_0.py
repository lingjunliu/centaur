
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_restore_slice_inputs():
    list_of_inputs = []

    # Input 1
    file_pattern = tf.constant(b"checkpoint_file")
    tensor_name = tf.constant(b"my_tensor")
    shape_and_slice = tf.constant(b"- 1 10,0 5")
    dt = tf.float32
    preferred_shard = -1
    name = "restore_slice_op_1"

    input_dict = {
        "file_pattern": file_pattern,
        "tensor_name": tensor_name,
        "shape_and_slice": shape_and_slice,
        "dt": dt,
        "preferred_shard": preferred_shard,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    file_pattern = tf.constant(b"another_checkpoint")
    tensor_name = tf.constant(b"another_tensor")
    shape_and_slice = tf.constant(b"- 5 20,2 8")
    dt = tf.int32
    preferred_shard = 0
    name = "restore_slice_op_2"

    input_dict = {
        "file_pattern": file_pattern,
        "tensor_name": tensor_name,
        "shape_and_slice": shape_and_slice,
        "dt": dt,
        "preferred_shard": preferred_shard,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    file_pattern = tf.constant(b"yet_another_checkpoint")
    tensor_name = tf.constant(b"yet_another_tensor")
    shape_and_slice = tf.constant(b"- 10 30,5 15")
    dt = tf.float64
    preferred_shard = 1
    name = "restore_slice_op_3"

    input_dict = {
        "file_pattern": file_pattern,
        "tensor_name": tensor_name,
        "shape_and_slice": shape_and_slice,
        "dt": dt,
        "preferred_shard": preferred_shard,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    file_pattern = tf.constant(b"checkpoint_file_4")
    tensor_name = tf.constant(b"my_tensor_4")
    shape_and_slice = tf.constant(b"- 2 4,0 2")
    dt = tf.int64
    preferred_shard = -1
    name = "restore_slice_op_4"

    input_dict = {
        "file_pattern": file_pattern,
        "tensor_name": tensor_name,
        "shape_and_slice": shape_and_slice,
        "dt": dt,
        "preferred_shard": preferred_shard,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    file_pattern = tf.constant(b"checkpoint_file_5")
    tensor_name = tf.constant(b"my_tensor_5")
    shape_and_slice = tf.constant(b"- 3 6,1 3")
    dt = tf.bool
    preferred_shard = 0
    name = "restore_slice_op_5"

    input_dict = {
        "file_pattern": file_pattern,
        "tensor_name": tensor_name,
        "shape_and_slice": shape_and_slice,
        "dt": dt,
        "preferred_shard": preferred_shard,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    file_pattern = tf.constant(b"checkpoint_file_6")
    tensor_name = tf.constant(b"my_tensor_6")
    shape_and_slice = tf.constant(b"- 4 8,2 4")
    dt = tf.complex64
    preferred_shard = 1
    name = "restore_slice_op_6"

    input_dict = {
        "file_pattern": file_pattern,
        "tensor_name": tensor_name,
        "shape_and_slice": shape_and_slice,
        "dt": dt,
        "preferred_shard": preferred_shard,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    file_pattern = tf.constant(b"checkpoint_file_7")
    tensor_name = tf.constant(b"my_tensor_7")
    shape_and_slice = tf.constant(b"- 5 10,3 5")
    dt = tf.complex128
    preferred_shard = -1
    name = "restore_slice_op_7"

    input_dict = {
        "file_pattern": file_pattern,
        "tensor_name": tensor_name,
        "shape_and_slice": shape_and_slice,
        "dt": dt,
        "preferred_shard": preferred_shard,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    file_pattern = tf.constant(b"checkpoint_file_8")
    tensor_name = tf.constant(b"my_tensor_8")
    shape_and_slice = tf.constant(b"- 6 12,4 6")
    dt = tf.uint8
    preferred_shard = 0
    name = "restore_slice_op_8"

    input_dict = {
        "file_pattern": file_pattern,
        "tensor_name": tensor_name,
        "shape_and_slice": shape_and_slice,
        "dt": dt,
        "preferred_shard": preferred_shard,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    file_pattern = tf.constant(b"checkpoint_file_9")
    tensor_name = tf.constant(b"my_tensor_9")
    shape_and_slice = tf.constant(b"- 7 14,5 7")
    dt = tf.int8
    preferred_shard = 1
    name = "restore_slice_op_9"

    input_dict = {
        "file_pattern": file_pattern,
        "tensor_name": tensor_name,
        "shape_and_slice": shape_and_slice,
        "dt": dt,
        "preferred_shard": preferred_shard,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    file_pattern = tf.constant(b"checkpoint_file_10")
    tensor_name = tf.constant(b"my_tensor_10")
    shape_and_slice = tf.constant(b"- 8 16,6 8")
    dt = tf.float16
    preferred_shard = -1
    name = "restore_slice_op_10"

    input_dict = {
        "file_pattern": file_pattern,
        "tensor_name": tensor_name,
        "shape_and_slice": shape_and_slice,
        "dt": dt,
        "preferred_shard": preferred_shard,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.RestoreSlice"] = tf_raw_ops_restore_slice_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.RestoreSlice' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RestoreSlice'.")

check_valid('tf.raw_ops.RestoreSlice', generated_inputs['tf.raw_ops.RestoreSlice'], lib="tf", suffix=0)
