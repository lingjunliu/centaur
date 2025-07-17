
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_BarrierIncompleteSize_inputs():
    list_of_inputs = []

    # Input 1
    handle = tf.constant(np.array("barrier_1", dtype=np.object_), dtype=tf.string)
    input_dict = {"handle": handle, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    handle = tf.constant(np.array("barrier_2", dtype=np.object_), dtype=tf.string)
    input_dict = {"handle": handle, "name": np.array("my_barrier_size", dtype=np.object_)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    handle = tf.constant(np.array("barrier_3", dtype=np.object_), dtype=tf.string)
    input_dict = {"handle": handle, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    handle = tf.constant(np.array("barrier_4_very_long_name", dtype=np.object_), dtype=tf.string)
    input_dict = {"handle": handle, "name": np.array("a_very_long_name_for_the_op", dtype=np.object_)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    handle = tf.constant(np.array("", dtype=np.object_), dtype=tf.string)
    input_dict = {"handle": handle, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    handle = tf.constant(np.array(" ", dtype=np.object_), dtype=tf.string)
    input_dict = {"handle": handle, "name": np.array("name_with_space", dtype=np.object_)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    handle = tf.constant(np.array("barrier_7", dtype=np.object_), dtype=tf.string)
    input_dict = {"handle": handle, "name": np.array("", dtype=np.object_)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    handle = tf.constant(np.array("barrier_8", dtype=np.object_), dtype=tf.string)
    input_dict = {"handle": handle, "name": np.array(" ", dtype=np.object_)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    handle = tf.constant(np.array("barrier_9", dtype=np.object_), dtype=tf.string)
    input_dict = {"handle": handle, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    handle = tf.constant(np.array("barrier_10", dtype=np.object_), dtype=tf.string)
    input_dict = {"handle": handle, "name": np.array("size_op", dtype=np.object_)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.BarrierIncompleteSize"] = tf_raw_ops_BarrierIncompleteSize_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.BarrierIncompleteSize' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.BarrierIncompleteSize'.")

check_valid('tf.raw_ops.BarrierIncompleteSize', generated_inputs['tf.raw_ops.BarrierIncompleteSize'], lib="tf", suffix=0)
