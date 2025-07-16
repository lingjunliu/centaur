
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_BarrierInsertMany_inputs():
    list_of_inputs = []

    # Input 1
    handle = tf.Variable("dummy_handle", dtype=tf.string)
    keys = tf.constant(["key1", "key2"], dtype=tf.string)
    values = tf.constant([[1, 2], [3, 4]], dtype=tf.int32)
    component_index = 0
    name = "insert_op_1"

    input_dict = {
        "handle": handle,
        "keys": keys,
        "values": values,
        "component_index": component_index,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    handle = tf.Variable("dummy_handle", dtype=tf.string)
    keys = tf.constant(["key3"], dtype=tf.string)
    values = tf.constant([[5, 6, 7]], dtype=tf.float32)
    component_index = 1
    name = "insert_op_2"

    input_dict = {
        "handle": handle,
        "keys": keys,
        "values": values,
        "component_index": component_index,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    handle = tf.Variable("dummy_handle", dtype=tf.string)
    keys = tf.constant(["key4", "key5", "key6"], dtype=tf.string)
    values = tf.constant([["a", "b"], ["c", "d"], ["e", "f"]], dtype=tf.string)
    component_index = 2
    name = "insert_op_3"

    input_dict = {
        "handle": handle,
        "keys": keys,
        "values": values,
        "component_index": component_index,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    handle = tf.Variable("dummy_handle", dtype=tf.string)
    keys = tf.constant(["key7"], dtype=tf.string)
    values = tf.constant([10], dtype=tf.int64)
    component_index = 3
    name = "insert_op_4"

    input_dict = {
        "handle": handle,
        "keys": keys,
        "values": values,
        "component_index": component_index,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    handle = tf.Variable("dummy_handle", dtype=tf.string)
    keys = tf.constant(["key8", "key9"], dtype=tf.string)
    values = tf.constant([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], dtype=tf.float64)
    component_index = 0
    name = "insert_op_5"

    input_dict = {
        "handle": handle,
        "keys": keys,
        "values": values,
        "component_index": component_index,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    handle = tf.Variable("dummy_handle", dtype=tf.string)
    keys = tf.constant(["key10"], dtype=tf.string)
    values = tf.constant([[[1, 2], [3, 4]]], dtype=tf.int32)
    component_index = 1
    name = "insert_op_6"

    input_dict = {
        "handle": handle,
        "keys": keys,
        "values": values,
        "component_index": component_index,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    handle = tf.Variable("dummy_handle", dtype=tf.string)
    keys = tf.constant(["key11", "key12"], dtype=tf.string)
    values = tf.constant([True, False], dtype=tf.bool)
    component_index = 2
    name = "insert_op_7"

    input_dict = {
        "handle": handle,
        "keys": keys,
        "values": values,
        "component_index": component_index,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    handle = tf.Variable("dummy_handle", dtype=tf.string)
    keys = tf.constant(["key13"], dtype=tf.string)
    values = tf.constant([12345], dtype=tf.int32)
    component_index = 0
    name = "insert_op_8"

    input_dict = {
        "handle": handle,
        "keys": keys,
        "values": values,
        "component_index": component_index,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    handle = tf.Variable("dummy_handle", dtype=tf.string)
    keys = tf.constant(["key14", "key15"], dtype=tf.string)
    values = tf.constant([[-1, -2], [-3, -4]], dtype=tf.int32)
    component_index = 1
    name = "insert_op_9"

    input_dict = {
        "handle": handle,
        "keys": keys,
        "values": values,
        "component_index": component_index,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    handle = tf.Variable("dummy_handle", dtype=tf.string)
    keys = tf.constant(["key16"], dtype=tf.string)
    values = tf.constant([[[1.1, 2.2], [3.3, 4.4]], [[5.5, 6.6], [7.7, 8.8]]], dtype=tf.float32)
    component_index = 2
    name = "insert_op_10"

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
generated_inputs["tf.raw_ops.BarrierInsertMany"] = tf_raw_ops_BarrierInsertMany_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.BarrierInsertMany' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.BarrierInsertMany'.")

check_valid('tf.raw_ops.BarrierInsertMany', generated_inputs['tf.raw_ops.BarrierInsertMany'], lib="tf", suffix=0)
