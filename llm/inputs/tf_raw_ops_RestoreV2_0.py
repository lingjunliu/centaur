
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_restore_v2_inputs():
    list_of_inputs = []

    # Input 1
    prefix = tf.constant("checkpoint_prefix", dtype=tf.string)
    tensor_names = tf.constant(["tensor1", "tensor2"], dtype=tf.string)
    shape_and_slices = tf.constant(["", ""], dtype=tf.string)
    dtypes = [tf.float32, tf.int32]
    input_dict = {
        "prefix": prefix,
        "tensor_names": tensor_names,
        "shape_and_slices": shape_and_slices,
        "dtypes": dtypes,
        "name": "restore_op_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    prefix = tf.constant("another_prefix", dtype=tf.string)
    tensor_names = tf.constant(["tensor_a", "tensor_b", "tensor_c"], dtype=tf.string)
    shape_and_slices = tf.constant(["", "-", "0:1,2:"], dtype=tf.string)
    dtypes = [tf.float64, tf.int64, tf.float16]
    input_dict = {
        "prefix": prefix,
        "tensor_names": tensor_names,
        "shape_and_slices": shape_and_slices,
        "dtypes": dtypes,
        "name": "restore_op_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    prefix = tf.constant("prefix_3", dtype=tf.string)
    tensor_names = tf.constant(["weight", "bias"], dtype=tf.string)
    shape_and_slices = tf.constant(["", ""], dtype=tf.string)
    dtypes = [tf.complex64, tf.bool]
    input_dict = {
        "prefix": prefix,
        "tensor_names": tensor_names,
        "shape_and_slices": shape_and_slices,
        "dtypes": dtypes,
        "name": "restore_op_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    prefix = tf.constant("prefix_4", dtype=tf.string)
    tensor_names = tf.constant(["var1"], dtype=tf.string)
    shape_and_slices = tf.constant(["0:10,20:30"], dtype=tf.string)
    dtypes = [tf.float32]
    input_dict = {
        "prefix": prefix,
        "tensor_names": tensor_names,
        "shape_and_slices": shape_and_slices,
        "dtypes": dtypes,
        "name": "restore_op_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 5
    prefix = tf.constant("prefix_5", dtype=tf.string)
    tensor_names = tf.constant(["tensor_x", "tensor_y"], dtype=tf.string)
    shape_and_slices = tf.constant(["", ""], dtype=tf.string)
    dtypes = [tf.qint8, tf.quint8]
    input_dict = {
        "prefix": prefix,
        "tensor_names": tensor_names,
        "shape_and_slices": shape_and_slices,
        "dtypes": dtypes,
        "name": "restore_op_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    prefix = tf.constant("prefix_6", dtype=tf.string)
    tensor_names = tf.constant(["single_tensor"], dtype=tf.string)
    shape_and_slices = tf.constant(["10:20"], dtype=tf.string)
    dtypes = [tf.float32]
    input_dict = {
        "prefix": prefix,
        "tensor_names": tensor_names,
        "shape_and_slices": shape_and_slices,
        "dtypes": dtypes,
        "name": "restore_op_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    prefix = tf.constant("prefix_7", dtype=tf.string)
    tensor_names = tf.constant(["tensor_z"], dtype=tf.string)
    shape_and_slices = tf.constant([""], dtype=tf.string)
    dtypes = [tf.float32]
    input_dict = {
        "prefix": prefix,
        "tensor_names": tensor_names,
        "shape_and_slices": shape_and_slices,
        "dtypes": dtypes,
        "name": "restore_op_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    prefix = tf.constant("prefix_8", dtype=tf.string)
    tensor_names = tf.constant(["sparse_tensor"], dtype=tf.string)
    shape_and_slices = tf.constant(["0:100"], dtype=tf.string)
    dtypes = [tf.float32] # Changed from tf.bfloat16
    input_dict = {
        "prefix": prefix,
        "tensor_names": tensor_names,
        "shape_and_slices": shape_and_slices,
        "dtypes": dtypes,
        "name": "restore_op_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    prefix = tf.constant("prefix_9", dtype=tf.string)
    tensor_names = tf.constant(["long_tensor"], dtype=tf.string)
    shape_and_slices = tf.constant([""], dtype=tf.string)
    dtypes = [tf.float32] # Changed from tf.uint8
    input_dict = {
        "prefix": prefix,
        "tensor_names": tensor_names,
        "shape_and_slices": shape_and_slices,
        "dtypes": dtypes,
        "name": "restore_op_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    prefix = tf.constant("prefix_10", dtype=tf.string)
    tensor_names = tf.constant(["name_only"], dtype=tf.string)
    shape_and_slices = tf.constant([""], dtype=tf.string)
    dtypes = [tf.float32]
    input_dict = {
        "prefix": prefix,
        "tensor_names": tensor_names,
        "shape_and_slices": shape_and_slices,
        "dtypes": dtypes,
        "name": "restore_op_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.RestoreV2"] = tf_raw_ops_restore_v2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.RestoreV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RestoreV2'.")

check_valid('tf.raw_ops.RestoreV2', generated_inputs['tf.raw_ops.RestoreV2'], lib="tf", suffix=0)
