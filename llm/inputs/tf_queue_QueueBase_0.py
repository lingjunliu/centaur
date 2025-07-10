
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_queue_queuebase_inputs():
    list_of_inputs = []

    # Input 1
    dtypes = [np.int32]
    shapes = [()]
    names = ['val']
    queue_ref = tf.constant("test_queue", dtype=tf.string)
    input_dict = {
        'dtypes': dtypes,
        'shapes': shapes,
        'names': names,
        'queue_ref': queue_ref
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    dtypes = [np.float32, np.int64]
    shapes = [(2,), ()]
    names = ['vec', 'scalar']
    queue_ref = tf.constant("another_queue", dtype=tf.string)
    input_dict = {
        'dtypes': dtypes,
        'shapes': shapes,
        'names': names,
        'queue_ref': queue_ref
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    dtypes = [np.bool_]
    shapes = [(3, 3)]
    names = ['matrix']
    queue_ref = tf.constant("bool_queue", dtype=tf.string)
    input_dict = {
        'dtypes': dtypes,
        'shapes': shapes,
        'names': names,
        'queue_ref': queue_ref
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    dtypes = [np.complex64]
    shapes = [()]
    names = ['complex_val']
    queue_ref = tf.constant("complex_queue", dtype=tf.string)
    input_dict = {
        'dtypes': dtypes,
        'shapes': shapes,
        'names': names,
        'queue_ref': queue_ref
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    dtypes = [np.int32, np.float64]
    shapes = [(), (5,)]
    names = ['int_val', 'float_vec']
    queue_ref = tf.constant("mixed_queue", dtype=tf.string)
    input_dict = {
        'dtypes': dtypes,
        'shapes': shapes,
        'names': names,
        'queue_ref': queue_ref
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    # Input 6
    dtypes = [np.int8, np.uint8]
    shapes = [(), ()]
    names = ['int8', 'uint8']
    queue_ref = tf.constant("int_queue", dtype=tf.string)
    input_dict = {
        'dtypes': dtypes,
        'shapes': shapes,
        'names': names,
        'queue_ref': queue_ref
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    dtypes = [np.int32]
    shapes = [(1, 2, 3, 4)]
    names = ['4d_tensor']
    queue_ref = tf.constant("4dqueue", dtype=tf.string)
    input_dict = {
        'dtypes': dtypes,
        'shapes': shapes,
        'names': names,
        'queue_ref': queue_ref
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.queue.QueueBase"] = tf_queue_queuebase_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.queue.QueueBase' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.queue.QueueBase'.")

check_valid('tf.queue.QueueBase', generated_inputs['tf.queue.QueueBase'], lib="tf", suffix=0)
