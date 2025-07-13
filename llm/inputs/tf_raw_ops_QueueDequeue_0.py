
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_QueueDequeue_inputs():
    list_of_inputs = []

    # Input 1
    handle = tf.constant("queue_handle", dtype=tf.string)
    component_types = [tf.float32.as_numpy_dtype]
    timeout_ms = -1
    name = "dequeue_op_1"
    input_dict = {"handle": handle, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    handle = tf.constant("another_queue", dtype=tf.string)
    component_types = [tf.int32.as_numpy_dtype]
    timeout_ms = 100
    name = "dequeue_op_2"
    input_dict = {"handle": handle, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    handle = tf.constant("queue3", dtype=tf.string)
    component_types = [tf.float32.as_numpy_dtype]
    timeout_ms = 0
    name = "dequeue_op_3"
    input_dict = {"handle": handle, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    handle = tf.constant("queue4", dtype=tf.string)
    component_types = [tf.bool.as_numpy_dtype]
    timeout_ms = -10
    name = None
    input_dict = {"handle": handle, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    handle = tf.constant("queue5", dtype=tf.string)
    component_types = [tf.uint8.as_numpy_dtype]
    timeout_ms = 1000
    name = "dequeue_op_5"
    input_dict = {"handle": handle, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    handle = tf.constant("queue6", dtype=tf.string)
    component_types = [tf.float16.as_numpy_dtype]
    timeout_ms = -1
    name = "dequeue_op_6"
    input_dict = {"handle": handle, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    handle = tf.constant("queue7", dtype=tf.string)
    component_types = [tf.complex64.as_numpy_dtype]
    timeout_ms = 50
    name = None
    input_dict = {"handle": handle, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    handle = tf.constant("queue8", dtype=tf.string)
    component_types = [tf.int64.as_numpy_dtype]
    timeout_ms = 200
    name = "dequeue_op_8"
    input_dict = {"handle": handle, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    handle = tf.constant("queue9", dtype=tf.string)
    component_types = [tf.float64.as_numpy_dtype]
    timeout_ms = -1
    name = "dequeue_op_9"
    input_dict = {"handle": handle, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    handle = tf.constant("queue10", dtype=tf.string)
    component_types = [tf.int8.as_numpy_dtype]
    timeout_ms = 1500
    name = None
    input_dict = {"handle": handle, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.QueueDequeue"] = tf_raw_ops_QueueDequeue_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.QueueDequeue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QueueDequeue'.")

check_valid('tf.raw_ops.QueueDequeue', generated_inputs['tf.raw_ops.QueueDequeue'], lib="tf", suffix=0)
