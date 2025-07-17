
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_BarrierInsertMany_inputs():
    list_of_inputs = []

    # Input 1
    handle = tf.constant("barrier_handle_1", dtype=tf.string)
    keys = tf.constant(["key1", "key2"], dtype=tf.string)
    values = tf.constant([[1, 2], [3, 4]], dtype=tf.int32)
    component_index = np.int32(0)
    input_dict = {"handle": handle, "keys": keys, "values": values, "component_index": component_index, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    handle = tf.constant("barrier_handle_2", dtype=tf.string)
    keys = tf.constant(["key3"], dtype=tf.string)
    values = tf.constant([[5.5, 6.6]], dtype=tf.float32)
    component_index = np.int32(1)
    input_dict = {"handle": handle, "keys": keys, "values": values, "component_index": component_index, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    handle = tf.constant("barrier_handle_3", dtype=tf.string)
    keys = tf.constant(["key4", "key5", "key6"], dtype=tf.string)
    values = tf.constant([["a", "b"], ["c", "d"], ["e", "f"]], dtype=tf.string)
    component_index = np.int32(2)
    input_dict = {"handle": handle, "keys": keys, "values": values, "component_index": component_index, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    handle = tf.constant("barrier_handle_4", dtype=tf.string)
    keys = tf.constant(["key7"], dtype=tf.string)
    values = tf.constant([[[1, 2, 3], [4, 5, 6]]], dtype=tf.int64)
    component_index = np.int32(3)
    input_dict = {"handle": handle, "keys": keys, "values": values, "component_index": component_index, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    handle = tf.constant("barrier_handle_5", dtype=tf.string)
    keys = tf.constant(["key8", "key9"], dtype=tf.string)
    values = tf.constant([[True, False], [False, True]], dtype=tf.bool)
    component_index = np.int32(4)
    input_dict = {"handle": handle, "keys": keys, "values": values, "component_index": component_index, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    handle = tf.constant("barrier_handle_6", dtype=tf.string)
    keys = tf.constant(["key10"], dtype=tf.string)
    values = tf.constant([[10]], dtype=tf.uint8)
    component_index = np.int32(5)
    input_dict = {"handle": handle, "keys": keys, "values": values, "component_index": component_index, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    handle = tf.constant("barrier_handle_7", dtype=tf.string)
    keys = tf.constant(["key11", "key12"], dtype=tf.string)
    values = tf.constant([[-1.0, -2.0], [-3.0, -4.0]], dtype=tf.float64)
    component_index = np.int32(6)
    input_dict = {"handle": handle, "keys": keys, "values": values, "component_index": component_index, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    handle = tf.constant("barrier_handle_8", dtype=tf.string)
    keys = tf.constant(["key13"], dtype=tf.string)
    values = tf.constant([[7,8,9,10]], dtype=tf.int16)
    component_index = np.int32(7)
    input_dict = {"handle": handle, "keys": keys, "values": values, "component_index": component_index, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9
    handle = tf.constant("barrier_handle_9", dtype=tf.string)
    keys = tf.constant(["key14", "key15", "key16", "key17"], dtype=tf.string)
    values = tf.constant([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=tf.complex64)
    component_index = np.int32(8)
    input_dict = {"handle": handle, "keys": keys, "values": values, "component_index": component_index, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    handle = tf.constant("barrier_handle_10", dtype=tf.string)
    keys = tf.constant(["key18"], dtype=tf.string)
    values = tf.constant([[[1.23456789], [9.87654321]]], dtype=tf.double)
    component_index = np.int32(9)
    input_dict = {"handle": handle, "keys": keys, "values": values, "component_index": component_index, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.BarrierInsertMany"] = tf_raw_ops_BarrierInsertMany_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.BarrierInsertMany' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.BarrierInsertMany'.")

check_valid('tf.raw_ops.BarrierInsertMany', generated_inputs['tf.raw_ops.BarrierInsertMany'], lib="tf", suffix=0)
