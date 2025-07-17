
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_restorev2_inputs():
    list_of_inputs = []

    # Input 1
    prefix = tf.constant("checkpoint_prefix_1", dtype=tf.string)
    tensor_names = tf.constant(["tensor_a", "tensor_b"], dtype=tf.string)
    shape_and_slices = tf.constant(["", ""], dtype=tf.string)
    dtypes = [tf.float32, tf.int32]

    input_dict = {
        "prefix": prefix,
        "tensor_names": tensor_names,
        "shape_and_slices": shape_and_slices,
        "dtypes": dtypes,
        "name": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    prefix = tf.constant("checkpoint_prefix_2", dtype=tf.string)
    tensor_names = tf.constant(["tensor_c"], dtype=tf.string)
    shape_and_slices = tf.constant([""], dtype=tf.string)
    dtypes = [tf.bool]

    input_dict = {
        "prefix": prefix,
        "tensor_names": tensor_names,
        "shape_and_slices": shape_and_slices,
        "dtypes": dtypes,
        "name": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    prefix = tf.constant("checkpoint_prefix_3", dtype=tf.string)
    tensor_names = tf.constant(["tensor_d", "tensor_e"], dtype=tf.string)
    shape_and_slices = tf.constant(["", ""], dtype=tf.string)
    dtypes = [tf.int64, tf.float64]

    input_dict = {
        "prefix": prefix,
        "tensor_names": tensor_names,
        "shape_and_slices": shape_and_slices,
        "dtypes": dtypes,
        "name": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    prefix = tf.constant("checkpoint_prefix_4", dtype=tf.string)
    tensor_names = tf.constant(["tensor_g"], dtype=tf.string)
    shape_and_slices = tf.constant([""], dtype=tf.string)
    dtypes = [tf.complex64]

    input_dict = {
        "prefix": prefix,
        "tensor_names": tensor_names,
        "shape_and_slices": shape_and_slices,
        "dtypes": dtypes,
        "name": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    prefix = tf.constant("checkpoint_prefix_5", dtype=tf.string)
    tensor_names = tf.constant(["tensor_h", "tensor_i"], dtype=tf.string)
    shape_and_slices = tf.constant(["", ""], dtype=tf.string)
    dtypes = [tf.uint8, tf.int16]

    input_dict = {
        "prefix": prefix,
        "tensor_names": tensor_names,
        "shape_and_slices": shape_and_slices,
        "dtypes": dtypes,
        "name": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6
    prefix = tf.constant("checkpoint_prefix_6", dtype=tf.string)
    tensor_names = tf.constant(["tensor_j"], dtype=tf.string)
    shape_and_slices = tf.constant([""], dtype=tf.string)
    dtypes = [tf.qint8]

    input_dict = {
        "prefix": prefix,
        "tensor_names": tensor_names,
        "shape_and_slices": shape_and_slices,
        "dtypes": dtypes,
        "name": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    prefix = tf.constant("checkpoint_prefix_7", dtype=tf.string)
    tensor_names = tf.constant(["tensor_k", "tensor_l"], dtype=tf.string)
    shape_and_slices = tf.constant(["", ""], dtype=tf.string)
    dtypes = [tf.quint8, tf.qint32]

    input_dict = {
        "prefix": prefix,
        "tensor_names": tensor_names,
        "shape_and_slices": shape_and_slices,
        "dtypes": dtypes,
        "name": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    prefix = tf.constant("checkpoint_prefix_8", dtype=tf.string)
    tensor_names = tf.constant(["tensor_m"], dtype=tf.string)
    shape_and_slices = tf.constant([""], dtype=tf.string)
    dtypes = [tf.resource]

    input_dict = {
        "prefix": prefix,
        "tensor_names": tensor_names,
        "shape_and_slices": shape_and_slices,
        "dtypes": dtypes,
        "name": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    prefix = tf.constant("checkpoint_prefix_9", dtype=tf.string)
    tensor_names = tf.constant(["tensor_n"], dtype=tf.string)
    shape_and_slices = tf.constant([""], dtype=tf.string)
    dtypes = [tf.variant]

    input_dict = {
        "prefix": prefix,
        "tensor_names": tensor_names,
        "shape_and_slices": shape_and_slices,
        "dtypes": dtypes,
        "name": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    prefix = tf.constant("checkpoint_prefix_10", dtype=tf.string)
    tensor_names = tf.constant(["tensor_o", "tensor_p"], dtype=tf.string)
    shape_and_slices = tf.constant([""], dtype=tf.string)
    dtypes = [tf.bfloat16, tf.float16]

    input_dict = {
        "prefix": prefix,
        "tensor_names": tensor_names,
        "shape_and_slices": shape_and_slices,
        "dtypes": dtypes,
        "name": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.RestoreV2"] = tf_raw_ops_restorev2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.RestoreV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RestoreV2'.")

check_valid('tf.raw_ops.RestoreV2', generated_inputs['tf.raw_ops.RestoreV2'], lib="tf", suffix=0)
