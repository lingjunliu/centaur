
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_queue_QueueBase_inputs():
    list_of_inputs = []

    # Input 1
    dtypes = [np.int32]
    shapes = [(10,)]
    names = ['data']
    queue_ref = tf.constant("queue1", dtype=tf.string)
    input_dict = {'dtypes': dtypes, 'shapes': shapes, 'names': names, 'queue_ref': queue_ref}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    dtypes = [np.float32, np.int64]
    shapes = [(), ()]
    names = ['value', 'index']
    queue_ref = tf.constant("queue2", dtype=tf.string)
    input_dict = {'dtypes': dtypes, 'shapes': shapes, 'names': names, 'queue_ref': queue_ref}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    dtypes = [np.string_, np.bool_]
    shapes = [(5, 5), (1,)]
    names = ['message', 'flag']
    queue_ref = tf.constant("queue3", dtype=tf.string)
    input_dict = {'dtypes': dtypes, 'shapes': shapes, 'names': names, 'queue_ref': queue_ref}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    dtypes = [np.complex64]
    shapes = [(2, 3, 4)]
    names = ['complex_data']
    queue_ref = tf.constant("queue4", dtype=tf.string)
    input_dict = {'dtypes': dtypes, 'shapes': shapes, 'names': names, 'queue_ref': queue_ref}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    dtypes = [np.uint8]
    shapes = [(10,)]
    names = ['byte_data']
    queue_ref = tf.constant("queue5", dtype=tf.string)
    input_dict = {'dtypes': dtypes, 'shapes': shapes, 'names': names, 'queue_ref': queue_ref}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    dtypes = [np.int32, np.float32, np.string_]
    shapes = [(1,), (2, 2), (3,)]
    names = ['id', 'matrix', 'label']
    queue_ref = tf.constant("queue6", dtype=tf.string)
    input_dict = {'dtypes': dtypes, 'shapes': shapes, 'names': names, 'queue_ref': queue_ref}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    dtypes = [np.int32]
    shapes = [()]
    names = ['scalar_data']
    queue_ref = tf.constant("queue7", dtype=tf.string)
    input_dict = {'dtypes': dtypes, 'shapes': shapes, 'names': names, 'queue_ref': queue_ref}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    dtypes = [np.float64, np.int8]
    shapes = [(1,1,1), (1,)]
    names = ['double_data', 'single_byte']
    queue_ref = tf.constant("queue8", dtype=tf.string)
    input_dict = {'dtypes': dtypes, 'shapes': shapes, 'names': names, 'queue_ref': queue_ref}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    dtypes = [np.bool_]
    shapes = [(4, 4)]
    names = ['boolean_matrix']
    queue_ref = tf.constant("queue9", dtype=tf.string)
    input_dict = {'dtypes': dtypes, 'shapes': shapes, 'names': names, 'queue_ref': queue_ref}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    dtypes = [np.int32, np.float32, np.string_, np.bool_]
    shapes = [(1,), (2, 2), (3,), (4,)]
    names = ['id', 'matrix', 'label', 'flag']
    queue_ref = tf.constant("queue10", dtype=tf.string)
    input_dict = {'dtypes': dtypes, 'shapes': shapes, 'names': names, 'queue_ref': queue_ref}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.queue.QueueBase"] = tf_queue_QueueBase_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.queue.QueueBase' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.queue.QueueBase'.")

check_valid('tf.queue.QueueBase', generated_inputs['tf.queue.QueueBase'], lib="tf", suffix=0)
