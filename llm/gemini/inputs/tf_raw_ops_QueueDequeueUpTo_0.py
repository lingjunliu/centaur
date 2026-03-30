
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_QueueDequeueUpTo_inputs():
    list_of_inputs = []

    # Input 1
    queue = tf.queue.FIFOQueue(10, dtypes=[tf.int32])
    handle = queue.queue_ref
    n = np.array(5, dtype=np.int32)
    component_types = [tf.dtypes.int32.as_numpy_dtype]
    timeout_ms = -1
    name = "dequeue_upto_1"
    input_dict = {"handle": handle, "n": n, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    queue = tf.queue.FIFOQueue(10, dtypes=[tf.float32, tf.int32])
    handle = queue.queue_ref
    n = np.array(3, dtype=np.int32)
    component_types = [tf.dtypes.float32.as_numpy_dtype, tf.dtypes.int32.as_numpy_dtype]
    timeout_ms = 100
    name = "dequeue_upto_2"
    input_dict = {"handle": handle, "n": n, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    queue = tf.queue.FIFOQueue(10, dtypes=[tf.string])
    handle = queue.queue_ref
    n = np.array(10, dtype=np.int32)
    component_types = [tf.dtypes.string.as_numpy_dtype]
    timeout_ms = -1
    name = "dequeue_upto_3"
    input_dict = {"handle": handle, "n": n, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    queue = tf.queue.FIFOQueue(10, dtypes=[tf.bool])
    handle = queue.queue_ref
    n = np.array(1, dtype=np.int32)
    component_types = [tf.dtypes.bool.as_numpy_dtype]
    timeout_ms = 50
    name = "dequeue_upto_4"
    input_dict = {"handle": handle, "n": n, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    queue = tf.queue.FIFOQueue(10, dtypes=[tf.int64])
    handle = queue.queue_ref
    n = np.array(7, dtype=np.int32)
    component_types = [tf.dtypes.int64.as_numpy_dtype]
    timeout_ms = -1
    name = "dequeue_upto_5"
    input_dict = {"handle": handle, "n": n, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    queue = tf.queue.FIFOQueue(10, dtypes=[tf.float64, tf.complex64])
    handle = queue.queue_ref
    n = np.array(2, dtype=np.int32)
    component_types = [tf.dtypes.float64.as_numpy_dtype, tf.dtypes.complex64.as_numpy_dtype]
    timeout_ms = 0
    name = "dequeue_upto_6"
    input_dict = {"handle": handle, "n": n, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    queue = tf.queue.FIFOQueue(10, dtypes=[tf.uint8])
    handle = queue.queue_ref
    n = np.array(4, dtype=np.int32)
    component_types = [tf.dtypes.uint8.as_numpy_dtype]
    timeout_ms = -1
    name = "dequeue_upto_7"
    input_dict = {"handle": handle, "n": n, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    queue = tf.queue.FIFOQueue(10, dtypes=[tf.int32, tf.int32, tf.int32])
    handle = queue.queue_ref
    n = np.array(6, dtype=np.int32)
    component_types = [tf.dtypes.int32.as_numpy_dtype, tf.dtypes.int32.as_numpy_dtype, tf.dtypes.int32.as_numpy_dtype]
    timeout_ms = -1
    name = "dequeue_upto_8"
    input_dict = {"handle": handle, "n": n, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9
    queue = tf.queue.FIFOQueue(10, dtypes=[tf.int32])
    handle = queue.queue_ref
    n = np.array(0, dtype=np.int32)
    component_types = [tf.dtypes.int32.as_numpy_dtype]
    timeout_ms = -1
    name = "dequeue_upto_9"
    input_dict = {"handle": handle, "n": n, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 10
    queue = tf.queue.FIFOQueue(10, dtypes=[tf.float32])
    handle = queue.queue_ref
    n = np.array(9, dtype=np.int32)
    component_types = [tf.dtypes.float32.as_numpy_dtype]
    timeout_ms = 200
    name = "dequeue_upto_10"
    input_dict = {"handle": handle, "n": n, "component_types": component_types, "timeout_ms": timeout_ms, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.QueueDequeueUpTo"] = tf_raw_ops_QueueDequeueUpTo_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.QueueDequeueUpTo' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QueueDequeueUpTo'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.QueueDequeueUpTo', generated_inputs['tf.raw_ops.QueueDequeueUpTo'], lib="tf", suffix=0)
