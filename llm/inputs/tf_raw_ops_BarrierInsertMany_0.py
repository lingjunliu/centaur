
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_barrier_insert_many_inputs():
    list_of_inputs = []

    # Input 1
    handle = tf.constant("barrier_handle", dtype=tf.string)
    keys = tf.constant(["key1", "key2"], dtype=tf.string)
    values = tf.constant([[1, 2], [3, 4]], dtype=tf.int32)
    component_index = 0
    name = None

    input_dict = {
        "handle": handle,
        "keys": keys,
        "values": values,
        "component_index": component_index,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    handle = tf.constant("another_handle", dtype=tf.string)
    keys = tf.constant(["key3"], dtype=tf.string)
    values = tf.constant([[5, 6, 7]], dtype=tf.float32)
    component_index = 1
    name = None

    input_dict = {
        "handle": handle,
        "keys": keys,
        "values": values,
        "component_index": component_index,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    handle = tf.constant("handle3", dtype=tf.string)
    keys = tf.constant(["key4", "key5", "key6"], dtype=tf.string)
    values = tf.constant([[8], [9], [10]], dtype=tf.int64)
    component_index = 2
    name = None

    input_dict = {
        "handle": handle,
        "keys": keys,
        "values": values,
        "component_index": component_index,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    handle = tf.constant("handle4", dtype=tf.string)
    keys = tf.constant(["key7"], dtype=tf.string)
    values = tf.constant([[11, 12, 13, 14]], dtype=tf.float32)
    component_index = 0
    name = None

    input_dict = {
        "handle": handle,
        "keys": keys,
        "values": values,
        "component_index": component_index,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    handle = tf.constant("handle5", dtype=tf.string)
    keys = tf.constant(["key8", "key9"], dtype=tf.string)
    values = tf.constant([[1, 0], [0, 1]], dtype=tf.int32)
    component_index = 1
    name = None

    input_dict = {
        "handle": handle,
        "keys": keys,
        "values": values,
        "component_index": component_index,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6
    handle = tf.constant("handle6", dtype=tf.string)
    keys = tf.constant(["key10"], dtype=tf.string)
    values = tf.constant([[15.5]], dtype=tf.float64)
    component_index = 2
    name = None

    input_dict = {
        "handle": handle,
        "keys": keys,
        "values": values,
        "component_index": component_index,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    handle = tf.constant("handle7", dtype=tf.string)
    keys = tf.constant(["key11", "key12"], dtype=tf.string)
    values = tf.constant([[16, 17, 18], [19, 20, 21]], dtype=tf.int16)
    component_index = 0
    name = None

    input_dict = {
        "handle": handle,
        "keys": keys,
        "values": values,
        "component_index": component_index,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    handle = tf.constant("handle8", dtype=tf.string)
    keys = tf.constant(["key13"], dtype=tf.string)
    values = tf.constant([[1]], dtype=tf.int32)
    component_index = 1
    name = None

    input_dict = {
        "handle": handle,
        "keys": keys,
        "values": values,
        "component_index": component_index,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    handle = tf.constant("handle9", dtype=tf.string)
    keys = tf.constant(["key14", "key15", "key16"], dtype=tf.string)
    values = tf.constant([[1,2,3]], dtype=tf.int8)
    component_index = 2
    name = None

    input_dict = {
        "handle": handle,
        "keys": keys,
        "values": values,
        "component_index": component_index,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    handle = tf.constant("handle10", dtype=tf.string)
    keys = tf.constant(["key17"], dtype=tf.string)
    values = tf.constant([[[4, 5], [6, 7]]], dtype=tf.int32)
    component_index = 0
    name = None

    input_dict = {
        "handle": handle,
        "keys": keys,
        "values": values,
        "component_index": component_index,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.BarrierInsertMany"] = tf_raw_ops_barrier_insert_many_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.BarrierInsertMany' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.BarrierInsertMany'.")

check_valid('tf.raw_ops.BarrierInsertMany', generated_inputs['tf.raw_ops.BarrierInsertMany'], lib="tf", suffix=0)
