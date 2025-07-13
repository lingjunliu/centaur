
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_BarrierTakeMany_inputs():
    list_of_inputs = []

    # Input 1
    handle = tf.constant("barrier_handle", dtype=tf.string)
    num_elements = tf.constant(2, dtype=tf.int32)
    component_types = [tf.float32]
    allow_small_batch = False
    wait_for_incomplete = False
    timeout_ms = -1
    name = "take_many_op_1"

    input_dict = {
        "handle": handle,
        "num_elements": num_elements,
        "component_types": component_types,
        "allow_small_batch": allow_small_batch,
        "wait_for_incomplete": wait_for_incomplete,
        "timeout_ms": timeout_ms,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    handle = tf.constant("another_handle", dtype=tf.string)
    num_elements = tf.constant(5, dtype=tf.int32)
    component_types = [tf.bool]
    allow_small_batch = True
    wait_for_incomplete = True
    timeout_ms = 100
    name = "take_many_op_2"

    input_dict = {
        "handle": handle,
        "num_elements": num_elements,
        "component_types": component_types,
        "allow_small_batch": allow_small_batch,
        "wait_for_incomplete": wait_for_incomplete,
        "timeout_ms": timeout_ms,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    handle = tf.constant("yet_another_handle", dtype=tf.string)
    num_elements = tf.constant(1, dtype=tf.int32)
    component_types = [tf.int64]
    allow_small_batch = False
    wait_for_incomplete = True
    timeout_ms = 0
    name = "take_many_op_3"

    input_dict = {
        "handle": handle,
        "num_elements": num_elements,
        "component_types": component_types,
        "allow_small_batch": allow_small_batch,
        "wait_for_incomplete": wait_for_incomplete,
        "timeout_ms": timeout_ms,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    handle = tf.constant("handle_4", dtype=tf.string)
    num_elements = tf.constant(10, dtype=tf.int32)
    component_types = [tf.float64]
    allow_small_batch = True
    wait_for_incomplete = False
    timeout_ms = -1
    name = "take_many_op_4"

    input_dict = {
        "handle": handle,
        "num_elements": num_elements,
        "component_types": component_types,
        "allow_small_batch": allow_small_batch,
        "wait_for_incomplete": wait_for_incomplete,
        "timeout_ms": timeout_ms,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    handle = tf.constant("handle_5", dtype=tf.string)
    num_elements = tf.constant(0, dtype=tf.int32)
    component_types = [tf.complex64]
    allow_small_batch = False
    wait_for_incomplete = True
    timeout_ms = 50
    name = "take_many_op_5"

    input_dict = {
        "handle": handle,
        "num_elements": num_elements,
        "component_types": component_types,
        "allow_small_batch": allow_small_batch,
        "wait_for_incomplete": wait_for_incomplete,
        "timeout_ms": timeout_ms,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6
    handle = tf.constant("handle_6", dtype=tf.string)
    num_elements = tf.constant(3, dtype=tf.int32)
    component_types = [tf.bfloat16]
    allow_small_batch = True
    wait_for_incomplete = True
    timeout_ms = -1
    name = "take_many_op_6"

    input_dict = {
        "handle": handle,
        "num_elements": num_elements,
        "component_types": component_types,
        "allow_small_batch": allow_small_batch,
        "wait_for_incomplete": wait_for_incomplete,
        "timeout_ms": timeout_ms,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    handle = tf.constant("handle_7", dtype=tf.string)
    num_elements = tf.constant(7, dtype=tf.int32)
    component_types = [tf.float32]
    allow_small_batch = False
    wait_for_incomplete = False
    timeout_ms = 1000
    name = "take_many_op_7"

    input_dict = {
        "handle": handle,
        "num_elements": num_elements,
        "component_types": component_types,
        "allow_small_batch": allow_small_batch,
        "wait_for_incomplete": wait_for_incomplete,
        "timeout_ms": timeout_ms,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    handle = tf.constant("handle_8", dtype=tf.string)
    num_elements = tf.constant(0, dtype=tf.int32)
    component_types = [tf.int32]
    allow_small_batch = True
    wait_for_incomplete = True
    timeout_ms = 200
    name = "take_many_op_8"

    input_dict = {
        "handle": handle,
        "num_elements": num_elements,
        "component_types": component_types,
        "allow_small_batch": allow_small_batch,
        "wait_for_incomplete": wait_for_incomplete,
        "timeout_ms": timeout_ms,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    handle = tf.constant("handle_9", dtype=tf.string)
    num_elements = tf.constant(1, dtype=tf.int32)
    component_types = [tf.float64]
    allow_small_batch = False
    wait_for_incomplete = False
    timeout_ms = -1
    name = "take_many_op_9"

    input_dict = {
        "handle": handle,
        "num_elements": num_elements,
        "component_types": component_types,
        "allow_small_batch": allow_small_batch,
        "wait_for_incomplete": wait_for_incomplete,
        "timeout_ms": timeout_ms,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    handle = tf.constant("handle_10", dtype=tf.string)
    num_elements = tf.constant(6, dtype=tf.int32)
    component_types = [tf.int32]
    allow_small_batch = True
    wait_for_incomplete = True
    timeout_ms = 500
    name = "take_many_op_10"

    input_dict = {
        "handle": handle,
        "num_elements": num_elements,
        "component_types": component_types,
        "allow_small_batch": allow_small_batch,
        "wait_for_incomplete": wait_for_incomplete,
        "timeout_ms": timeout_ms,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.BarrierTakeMany"] = tf_raw_ops_BarrierTakeMany_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.BarrierTakeMany' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.BarrierTakeMany'.")

check_valid('tf.raw_ops.BarrierTakeMany', generated_inputs['tf.raw_ops.BarrierTakeMany'], lib="tf", suffix=0)
