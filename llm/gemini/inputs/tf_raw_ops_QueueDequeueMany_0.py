
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_queue_dequeue_many_inputs():
    list_of_inputs = []

    # Input 1
    handle = tf.constant("dummy_handle", dtype=tf.string)
    n = tf.constant(2, dtype=tf.int32)
    component_types = [tf.float32]
    timeout_ms = -1
    name = None
    input_dict = {"handle": handle, "n": n, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    handle = tf.constant("another_dummy_handle", dtype=tf.string)
    n = tf.constant(5, dtype=tf.int32)
    component_types = [tf.int32, tf.float64]
    timeout_ms = 100
    name = "dequeue_op"
    input_dict = {"handle": handle, "n": n, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    handle = tf.constant("yet_another_dummy", dtype=tf.string)
    n = tf.constant(1, dtype=tf.int32)
    component_types = [tf.float32]
    timeout_ms = 0
    name = "string_dequeue"
    input_dict = {"handle": handle, "n": n, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    handle = tf.constant("handle_4", dtype=tf.string)
    n = tf.constant(10, dtype=tf.int32)
    component_types = [tf.bool]
    timeout_ms = -1
    name = None
    input_dict = {"handle": handle, "n": n, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    handle = tf.constant("handle_5", dtype=tf.string)
    n = tf.constant(3, dtype=tf.int32)
    component_types = [tf.int8, tf.int16, tf.int32]
    timeout_ms = 50
    name = "multi_type_dequeue"
    input_dict = {"handle": handle, "n": n, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    handle = tf.constant("handle_6", dtype=tf.string)
    n = tf.constant(7, dtype=tf.int32)
    component_types = [tf.uint8]
    timeout_ms = 200
    name = "uint8_dequeue"
    input_dict = {"handle": handle, "n": n, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    handle = tf.constant("handle_7", dtype=tf.string)
    n = tf.constant(4, dtype=tf.int32)
    component_types = [tf.complex64]
    timeout_ms = -1
    name = "complex_dequeue"
    input_dict = {"handle": handle, "n": n, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8
    handle = tf.constant("handle_8", dtype=tf.string)
    n = tf.constant(6, dtype=tf.int32)
    component_types = [tf.float16]
    timeout_ms = -1
    name = None
    input_dict = {"handle": handle, "n": n, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    handle = tf.constant("handle_9", dtype=tf.string)
    n = tf.constant(2, dtype=tf.int32)
    component_types = [tf.int64]
    timeout_ms = -1
    name = "variant_dequeue"
    input_dict = {"handle": handle, "n": n, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    handle = tf.constant("handle_10", dtype=tf.string)
    n = tf.constant(8, dtype=tf.int32)
    component_types = [tf.bfloat16]
    timeout_ms = -1
    name = None
    input_dict = {"handle": handle, "n": n, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.QueueDequeueMany"] = tf_raw_ops_queue_dequeue_many_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.QueueDequeueMany' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QueueDequeueMany'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.QueueDequeueMany', generated_inputs['tf.raw_ops.QueueDequeueMany'], lib="tf", suffix=0)
