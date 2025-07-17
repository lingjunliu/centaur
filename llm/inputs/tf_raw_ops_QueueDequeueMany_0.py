
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_QueueDequeueMany_inputs():
    list_of_inputs = []

    # Input 1
    handle = tf.constant("queue_handle", dtype=tf.string)
    n = tf.constant(2, dtype=tf.int32)
    component_types = [tf.int32, tf.float32]
    timeout_ms = -1
    name = "dequeue_many_1"
    input_dict = {"handle": handle, "n": n, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    handle = tf.constant("another_queue", dtype=tf.string)
    n = tf.constant(5, dtype=tf.int32)
    component_types = [tf.int32]
    timeout_ms = 100
    name = "dequeue_many_2"
    input_dict = {"handle": handle, "n": n, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    handle = tf.constant("yet_another_queue", dtype=tf.string)
    n = tf.constant(1, dtype=tf.int32)
    component_types = [tf.int64, tf.bool]
    timeout_ms = -1
    name = None
    input_dict = {"handle": handle, "n": n, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    handle = tf.constant("a_queue", dtype=tf.string)
    n = tf.constant(10, dtype=tf.int32)
    component_types = [tf.complex64]
    timeout_ms = 500
    name = "dequeue_complex"
    input_dict = {"handle": handle, "n": n, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    handle = tf.constant("special_queue", dtype=tf.string)
    n = tf.constant(0, dtype=tf.int32)
    component_types = [tf.int32]
    timeout_ms = -1
    name = "dequeue_zero"
    input_dict = {"handle": handle, "n": n, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    handle = tf.constant("mixed_queue", dtype=tf.string)
    n = tf.constant(3, dtype=tf.int32)
    component_types = [tf.int32, tf.float32]
    timeout_ms = 200
    name = "dequeue_mixed"
    input_dict = {"handle": handle, "n": n, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    handle = tf.constant("wide_queue", dtype=tf.string)
    n = tf.constant(4, dtype=tf.int32)
    component_types = [tf.float16]
    timeout_ms = -1
    name = None
    input_dict = {"handle": handle, "n": n, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    handle = tf.constant("large_n_queue", dtype=tf.string)
    n = tf.constant(100, dtype=tf.int32)
    component_types = [tf.int8]
    timeout_ms = 1000
    name = "large_n_dequeue"
    input_dict = {"handle": handle, "n": n, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    handle = tf.constant("empty_name_queue", dtype=tf.string)
    n = tf.constant(7, dtype=tf.int32)
    component_types = [tf.double]
    timeout_ms = -1
    name = ""
    input_dict = {"handle": handle, "n": n, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    handle = tf.constant("long_timeout", dtype=tf.string)
    n = tf.constant(2, dtype=tf.int32)
    component_types = [tf.uint8]
    timeout_ms = 60000
    name = "long_timeout_queue"
    input_dict = {"handle": handle, "n": n, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
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
