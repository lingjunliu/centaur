
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_barrier_take_many_inputs():
    list_of_inputs = []

    # Input 1
    handle = tf.constant("test_barrier", dtype=tf.string)
    num_elements = tf.constant(1, dtype=tf.int32)
    component_types = [tf.float32]
    allow_small_batch = False
    wait_for_incomplete = False
    timeout_ms = -1

    input_dict = {
        "handle": handle,
        "num_elements": num_elements,
        "component_types": component_types,
        "allow_small_batch": allow_small_batch,
        "wait_for_incomplete": wait_for_incomplete,
        "timeout_ms": timeout_ms,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    handle = tf.constant("another_barrier", dtype=tf.string)
    num_elements = tf.constant(5, dtype=tf.int32)
    component_types = [tf.int32]
    allow_small_batch = True
    wait_for_incomplete = True
    timeout_ms = 100

    input_dict = {
        "handle": handle,
        "num_elements": num_elements,
        "component_types": component_types,
        "allow_small_batch": allow_small_batch,
        "wait_for_incomplete": wait_for_incomplete,
        "timeout_ms": timeout_ms,
        "name": "take_many"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    handle = tf.constant("yet_another", dtype=tf.string)
    num_elements = tf.constant(10, dtype=tf.int32)
    component_types = [tf.float64]
    allow_small_batch = False
    wait_for_incomplete = True
    timeout_ms = 0

    input_dict = {
        "handle": handle,
        "num_elements": num_elements,
        "component_types": component_types,
        "allow_small_batch": allow_small_batch,
        "wait_for_incomplete": wait_for_incomplete,
        "timeout_ms": timeout_ms,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    handle = tf.constant("barrier_number_4", dtype=tf.string)
    num_elements = tf.constant(0, dtype=tf.int32)
    component_types = [tf.uint8]
    allow_small_batch = True
    wait_for_incomplete = False
    timeout_ms = -1

    input_dict = {
        "handle": handle,
        "num_elements": num_elements,
        "component_types": component_types,
        "allow_small_batch": allow_small_batch,
        "wait_for_incomplete": wait_for_incomplete,
        "timeout_ms": timeout_ms,
        "name": "zero_elements"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    handle = tf.constant("barrier_5", dtype=tf.string)
    num_elements = tf.constant(2, dtype=tf.int32)
    component_types = [tf.complex64]
    allow_small_batch = False
    wait_for_incomplete = False
    timeout_ms = 50

    input_dict = {
        "handle": handle,
        "num_elements": num_elements,
        "component_types": component_types,
        "allow_small_batch": allow_small_batch,
        "wait_for_incomplete": wait_for_incomplete,
        "timeout_ms": timeout_ms,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    handle = tf.constant("barrier_6", dtype=tf.string)
    num_elements = tf.constant(15, dtype=tf.int32)
    component_types = [tf.bool]
    allow_small_batch = True
    wait_for_incomplete = True
    timeout_ms = 1000

    input_dict = {
        "handle": handle,
        "num_elements": num_elements,
        "component_types": component_types,
        "allow_small_batch": allow_small_batch,
        "wait_for_incomplete": wait_for_incomplete,
        "timeout_ms": timeout_ms,
        "name": "resource_variant"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7
    handle = tf.constant("barrier_7", dtype=tf.string)
    num_elements = tf.constant(-1, dtype=tf.int32)
    component_types = [tf.bfloat16]
    allow_small_batch = False
    wait_for_incomplete = False
    timeout_ms = -1

    input_dict = {
        "handle": handle,
        "num_elements": num_elements,
        "component_types": component_types,
        "allow_small_batch": allow_small_batch,
        "wait_for_incomplete": wait_for_incomplete,
        "timeout_ms": timeout_ms,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    handle = tf.constant("barrier_8", dtype=tf.string)
    num_elements = tf.constant(1, dtype=tf.int32)
    component_types = [tf.float32]
    allow_small_batch = True
    wait_for_incomplete = True
    timeout_ms = 100

    input_dict = {
        "handle": handle,
        "num_elements": num_elements,
        "component_types": component_types,
        "allow_small_batch": allow_small_batch,
        "wait_for_incomplete": wait_for_incomplete,
        "timeout_ms": timeout_ms,
        "name": "take_many"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    handle = tf.constant("barrier_9", dtype=tf.string)
    num_elements = tf.constant(np.iinfo(np.int32).max, dtype=tf.int32)
    component_types = [tf.float64]
    allow_small_batch = False
    wait_for_incomplete = True
    timeout_ms = 0

    input_dict = {
        "handle": handle,
        "num_elements": num_elements,
        "component_types": component_types,
        "allow_small_batch": allow_small_batch,
        "wait_for_incomplete": wait_for_incomplete,
        "timeout_ms": timeout_ms,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    handle = tf.constant("barrier_10", dtype=tf.string)
    num_elements = tf.constant(np.iinfo(np.int32).min, dtype=tf.int32)
    component_types = [tf.uint8]
    allow_small_batch = True
    wait_for_incomplete = False
    timeout_ms = -1

    input_dict = {
        "handle": handle,
        "num_elements": num_elements,
        "component_types": component_types,
        "allow_small_batch": allow_small_batch,
        "wait_for_incomplete": wait_for_incomplete,
        "timeout_ms": timeout_ms,
        "name": "zero_elements"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
        
    # Input 11
    handle = tf.constant("barrier_11", dtype=tf.string)
    num_elements = tf.constant(1, dtype=tf.int32)
    component_types = [tf.resource]
    allow_small_batch = False
    wait_for_incomplete = False
    timeout_ms = -1

    input_dict = {
        "handle": handle,
        "num_elements": num_elements,
        "component_types": component_types,
        "allow_small_batch": allow_small_batch,
        "wait_for_incomplete": wait_for_incomplete,
        "timeout_ms": timeout_ms,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12
    handle = tf.constant("barrier_12", dtype=tf.string)
    num_elements = tf.constant(1, dtype=tf.int32)
    component_types = [tf.variant]
    allow_small_batch = False
    wait_for_incomplete = False
    timeout_ms = -1

    input_dict = {
        "handle": handle,
        "num_elements": num_elements,
        "component_types": component_types,
        "allow_small_batch": allow_small_batch,
        "wait_for_incomplete": wait_for_incomplete,
        "timeout_ms": timeout_ms,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 13
    handle = tf.constant("barrier_13", dtype=tf.string)
    num_elements = tf.constant(1, dtype=tf.int32)
    component_types = [tf.qint8]
    allow_small_batch = False
    wait_for_incomplete = False
    timeout_ms = -1

    input_dict = {
        "handle": handle,
        "num_elements": num_elements,
        "component_types": component_types,
        "allow_small_batch": allow_small_batch,
        "wait_for_incomplete": wait_for_incomplete,
        "timeout_ms": timeout_ms,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 14
    handle = tf.constant("barrier_14", dtype=tf.string)
    num_elements = tf.constant(1, dtype=tf.int32)
    component_types = [tf.quint8]
    allow_small_batch = False
    wait_for_incomplete = False
    timeout_ms = -1

    input_dict = {
        "handle": handle,
        "num_elements": num_elements,
        "component_types": component_types,
        "allow_small_batch": allow_small_batch,
        "wait_for_incomplete": wait_for_incomplete,
        "timeout_ms": timeout_ms,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 15
    handle = tf.constant("barrier_15", dtype=tf.string)
    num_elements = tf.constant(1, dtype=tf.int32)
    component_types = [tf.qint32]
    allow_small_batch = False
    wait_for_incomplete = False
    timeout_ms = -1

    input_dict = {
        "handle": handle,
        "num_elements": num_elements,
        "component_types": component_types,
        "allow_small_batch": allow_small_batch,
        "wait_for_incomplete": wait_for_incomplete,
        "timeout_ms": timeout_ms,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.BarrierTakeMany"] = tf_raw_ops_barrier_take_many_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.BarrierTakeMany' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.BarrierTakeMany'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.BarrierTakeMany', generated_inputs['tf.raw_ops.BarrierTakeMany'], lib="tf", suffix=0)
