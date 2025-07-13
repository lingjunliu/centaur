
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_QueueDequeueMany_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    handle = tf.constant("queue_handle", dtype=tf.string)
    n = tf.constant(2, dtype=tf.int32)
    component_types = [tf.float32, tf.int32]
    timeout_ms = -1
    name = "dequeue_many_op1"

    input_dict = {
        "handle": handle,
        "n": n,
        "component_types": component_types,
        "timeout_ms": timeout_ms,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different n value
    handle = tf.constant("queue_handle", dtype=tf.string)
    n = tf.constant(5, dtype=tf.int32)
    component_types = [tf.float32, tf.int32]
    timeout_ms = -1
    name = "dequeue_many_op2"
    input_dict = {
        "handle": handle,
        "n": n,
        "component_types": component_types,
        "timeout_ms": timeout_ms,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Single component type
    handle = tf.constant("queue_handle", dtype=tf.string)
    n = tf.constant(3, dtype=tf.int32)
    component_types = [tf.float32]
    timeout_ms = -1
    name = "dequeue_many_op3"
    input_dict = {
        "handle": handle,
        "n": n,
        "component_types": component_types,
        "timeout_ms": timeout_ms,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Timeout specified (still -1 for now due to lack of support)
    handle = tf.constant("queue_handle", dtype=tf.string)
    n = tf.constant(1, dtype=tf.int32)
    component_types = [tf.float64]
    timeout_ms = 100
    name = "dequeue_many_op4"
    input_dict = {
        "handle": handle,
        "n": n,
        "component_types": component_types,
        "timeout_ms": timeout_ms,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5: More complex component types
    handle = tf.constant("queue_handle", dtype=tf.string)
    n = tf.constant(2, dtype=tf.int32)
    component_types = [tf.complex64, tf.bool]
    timeout_ms = -1
    name = "dequeue_many_op5"
    input_dict = {
        "handle": handle,
        "n": n,
        "component_types": component_types,
        "timeout_ms": timeout_ms,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different queue handle name
    handle = tf.constant("another_queue", dtype=tf.string)
    n = tf.constant(4, dtype=tf.int32)
    component_types = [tf.int64]
    timeout_ms = -1
    name = "dequeue_many_op6"
    input_dict = {
        "handle": handle,
        "n": n,
        "component_types": component_types,
        "timeout_ms": timeout_ms,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Keep it simple
    handle = tf.constant("queue_handle", dtype=tf.string)
    n = tf.constant(2, dtype=tf.int32)
    component_types = [tf.uint8]
    timeout_ms = -1
    name = "dequeue_many_op7"
    input_dict = {
        "handle": handle,
        "n": n,
        "component_types": component_types,
        "timeout_ms": timeout_ms,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Keep it simple
    handle = tf.constant("queue_handle", dtype=tf.string)
    n = tf.constant(2, dtype=tf.int32)
    component_types = [tf.float16]
    timeout_ms = -1
    name = "dequeue_many_op8"
    input_dict = {
        "handle": handle,
        "n": n,
        "component_types": component_types,
        "timeout_ms": timeout_ms,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: name = None
    handle = tf.constant("queue_handle", dtype=tf.string)
    n = tf.constant(2, dtype=tf.int32)
    component_types = [tf.float32, tf.int32]
    timeout_ms = -1
    name = None

    input_dict = {
        "handle": handle,
        "n": n,
        "component_types": component_types,
        "timeout_ms": timeout_ms,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Keep it simple
    handle = tf.constant("another_queue", dtype=tf.string)
    n = tf.constant(2, dtype=tf.int32)
    component_types = [tf.int16]
    timeout_ms = 50
    name = "dequeue_many_op10"

    input_dict = {
        "handle": handle,
        "n": n,
        "component_types": component_types,
        "timeout_ms": timeout_ms,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.QueueDequeueMany"] = tf_raw_ops_QueueDequeueMany_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.QueueDequeueMany' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QueueDequeueMany'.")

check_valid('tf.raw_ops.QueueDequeueMany', generated_inputs['tf.raw_ops.QueueDequeueMany'], lib="tf", suffix=0)
