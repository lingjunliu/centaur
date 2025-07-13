
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_QueueDequeueUpTo_inputs():
    list_of_inputs = []

    # Input 1
    handle = tf.constant("dummy_handle", dtype=tf.string)
    n = tf.constant(5, dtype=tf.int32)
    component_types = [tf.float32]
    timeout_ms = -1
    name = "dequeue_up_to_1"
    input_dict = {"handle": handle, "n": n, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    handle = tf.constant("dummy_handle", dtype=tf.string)
    n = tf.constant(10, dtype=tf.int32)
    component_types = [tf.int32, tf.float32]
    timeout_ms = 100
    name = "dequeue_up_to_2"
    input_dict = {"handle": handle, "n": n, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    handle = tf.constant("dummy_handle", dtype=tf.string)
    n = tf.constant(1, dtype=tf.int32)
    component_types = [tf.string]
    timeout_ms = 0
    name = "dequeue_up_to_3"
    input_dict = {"handle": handle, "n": n, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    handle = tf.constant("dummy_handle", dtype=tf.string)
    n = tf.constant(20, dtype=tf.int32)
    component_types = [tf.bool, tf.int64, tf.float64]
    timeout_ms = -1
    name = "dequeue_up_to_4"
    input_dict = {"handle": handle, "n": n, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 5
    handle = tf.constant("dummy_handle", dtype=tf.string)
    n = tf.constant(0, dtype=tf.int32)
    component_types = [tf.float32]
    timeout_ms = -1
    name = "dequeue_up_to_5"
    input_dict = {"handle": handle, "n": n, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    handle = tf.constant("dummy_handle", dtype=tf.string)
    n = tf.constant(100, dtype=tf.int32)
    component_types = [tf.qint8, tf.quint8, tf.qint32]
    timeout_ms = 500
    name = "dequeue_up_to_6"
    input_dict = {"handle": handle, "n": n, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    handle = tf.constant("dummy_handle", dtype=tf.string)
    n = tf.constant(2, dtype=tf.int32)
    component_types = [tf.resource, tf.variant]
    timeout_ms = 10
    name = "dequeue_up_to_7"
    input_dict = {"handle": handle, "n": n, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    handle = tf.constant("dummy_handle", dtype=tf.string)
    n = tf.constant(3, dtype=tf.int32)
    component_types = [tf.complex64, tf.complex128]
    timeout_ms = -1
    name = "dequeue_up_to_8"
    input_dict = {"handle": handle, "n": n, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    handle = tf.constant("dummy_handle", dtype=tf.string)
    n = tf.constant(4, dtype=tf.int32)
    component_types = [tf.half, tf.bfloat16]
    timeout_ms = 1000
    name = "dequeue_up_to_9"
    input_dict = {"handle": handle, "n": n, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    handle = tf.constant("dummy_handle", dtype=tf.string)
    n = tf.constant(-1, dtype=tf.int32)
    component_types = [tf.int8]
    timeout_ms = -1
    name = "dequeue_up_to_10"
    input_dict = {"handle": handle, "n": n, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    handle = tf.constant("dummy_handle", dtype=tf.string)
    n = tf.constant(2147483647, dtype=tf.int32)
    component_types = [tf.uint8, tf.uint16, tf.uint32, tf.uint64]
    timeout_ms = -1
    name = "dequeue_up_to_11"
    input_dict = {"handle": handle, "n": n, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.QueueDequeueUpTo"] = tf_raw_ops_QueueDequeueUpTo_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.QueueDequeueUpTo' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QueueDequeueUpTo'.")

check_valid('tf.raw_ops.QueueDequeueUpTo', generated_inputs['tf.raw_ops.QueueDequeueUpTo'], lib="tf", suffix=0)
