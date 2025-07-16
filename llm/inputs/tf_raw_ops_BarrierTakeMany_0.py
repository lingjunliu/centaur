
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
    num_elements = tf.constant(1, dtype=tf.int32)
    component_types = [tf.float32]
    allow_small_batch = False
    wait_for_incomplete = False
    timeout_ms = -1
    name = "take_many_1"

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
    handle = tf.constant("another_barrier", dtype=tf.string)
    num_elements = tf.constant(5, dtype=tf.int32)
    component_types = [tf.int32]
    allow_small_batch = True
    wait_for_incomplete = True
    timeout_ms = 100
    name = "take_many_2"

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
    handle = tf.constant("test_barrier", dtype=tf.string)
    num_elements = tf.constant(10, dtype=tf.int32)
    component_types = [tf.float64]
    allow_small_batch = False
    wait_for_incomplete = True
    timeout_ms = 0
    name = "take_many_3"

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
    handle = tf.constant("empty_barrier", dtype=tf.string)
    num_elements = tf.constant(0, dtype=tf.int32)
    component_types = [tf.bool]
    allow_small_batch = True
    wait_for_incomplete = False
    timeout_ms = -1
    name = "take_many_4"

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
    handle = tf.constant("barrier5", dtype=tf.string)
    num_elements = tf.constant(2, dtype=tf.int32)
    component_types = [tf.int64]
    allow_small_batch = False
    wait_for_incomplete = False
    timeout_ms = 50
    name = "take_many_5"

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
    handle = tf.constant("barrier6", dtype=tf.string)
    num_elements = tf.constant(1, dtype=tf.int32)
    component_types = [tf.float32]
    allow_small_batch = False
    wait_for_incomplete = False
    timeout_ms = 50
    name = "take_many_6"

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
    handle = tf.constant("barrier7", dtype=tf.string)
    num_elements = tf.constant(1, dtype=tf.int32)
    component_types = [tf.float16]
    allow_small_batch = True
    wait_for_incomplete = True
    timeout_ms = 1000
    name = "take_many_7"

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
    handle = tf.constant("barrier8", dtype=tf.string)
    num_elements = tf.constant(1, dtype=tf.int32)
    component_types = [tf.complex64]
    allow_small_batch = False
    wait_for_incomplete = False
    timeout_ms = -1
    name = "take_many_8"

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
    handle = tf.constant("barrier9", dtype=tf.string)
    num_elements = tf.constant(1, dtype=tf.int32)
    component_types = [tf.resource]
    allow_small_batch = False
    wait_for_incomplete = False
    timeout_ms = -1
    name = "take_many_9"

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
    handle = tf.constant("barrier10", dtype=tf.string)
    num_elements = tf.constant(1, dtype=tf.int32)
    component_types = [tf.variant]
    allow_small_batch = False
    wait_for_incomplete = False
    timeout_ms = -1
    name = "take_many_10"

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
