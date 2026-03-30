
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_QueueIsClosed_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    q = tf.queue.FIFOQueue(capacity=10, dtypes=[tf.int32])
    handle = q.queue_ref
    with tf.compat.v1.Session() as sess:
        input_dict = {"handle": sess.run(handle), "name": "queue_is_closed_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different name
    q = tf.queue.FIFOQueue(capacity=5, dtypes=[tf.float32])
    handle = q.queue_ref
    with tf.compat.v1.Session() as sess:
        input_dict = {"handle": sess.run(handle), "name": "different_queue_name"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Empty name
    q = tf.queue.FIFOQueue(capacity=2, dtypes=[tf.string])
    handle = q.queue_ref
    with tf.compat.v1.Session() as sess:
        input_dict = {"handle": sess.run(handle), "name": ""}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Queue with different type
    q = tf.queue.FIFOQueue(capacity=3, dtypes=[tf.bool])
    handle = q.queue_ref
    with tf.compat.v1.Session() as sess:
        input_dict = {"handle": sess.run(handle), "name": "bool_queue"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Shared name
    q = tf.queue.FIFOQueue(capacity=7, dtypes=[tf.int64])
    handle = q.queue_ref
    with tf.compat.v1.Session() as sess:
        input_dict = {"handle": sess.run(handle), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Another shared name
    q = tf.queue.FIFOQueue(capacity=8, dtypes=[tf.float64])
    handle = q.queue_ref
    with tf.compat.v1.Session() as sess:
        input_dict = {"handle": sess.run(handle), "name": "another_shared_name"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Queue with larger capacity
    q = tf.queue.FIFOQueue(capacity=100, dtypes=[tf.int32])
    handle = q.queue_ref
    with tf.compat.v1.Session() as sess:
        input_dict = {"handle": sess.run(handle), "name": "larger_capacity_queue"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Queue with small capacity
    q = tf.queue.FIFOQueue(capacity=1, dtypes=[tf.int32])
    handle = q.queue_ref
    with tf.compat.v1.Session() as sess:
        input_dict = {"handle": sess.run(handle), "name": "smaller_capacity_queue"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Different queue type (PaddingFIFOQueue)
    q = tf.queue.PaddingFIFOQueue(capacity=5, dtypes=[tf.float32], shapes=[()])
    handle = q.queue_ref
    with tf.compat.v1.Session() as sess:
        input_dict = {"handle": sess.run(handle), "name": "padding_queue"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different queue type (RandomShuffleQueue)
    q = tf.queue.RandomShuffleQueue(capacity=5, min_after_dequeue=1, dtypes=[tf.float32])
    handle = q.queue_ref
    with tf.compat.v1.Session() as sess:
        input_dict = {"handle": sess.run(handle), "name": "random_queue"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.QueueIsClosed"] = tf_raw_ops_QueueIsClosed_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.QueueIsClosed' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QueueIsClosed'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.QueueIsClosed', generated_inputs['tf.raw_ops.QueueIsClosed'], lib="tf", suffix=0)
