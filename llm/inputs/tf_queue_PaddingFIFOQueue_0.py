
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_queue_paddingFIFOQueue_inputs():
    list_of_inputs = []

    # Input 1
    capacity = 5
    dtypes = [np.float32]
    shapes = [[None, 10]]
    names = [None] * 1 if [None] else None
    shared_name = None
    name = 'padding_fifo_queue_1'
    input_dict = {'capacity': capacity, 'dtypes': dtypes, 'shapes': shapes, 'names': names, 'shared_name': shared_name, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    capacity = 10
    dtypes = [np.int32, np.float64]
    shapes = [[None], [3, None]]
    names = ['int_data', 'float_data']
    shared_name = 'shared_queue_2'
    name = 'padding_fifo_queue_2'
    input_dict = {'capacity': capacity, 'dtypes': dtypes, 'shapes': shapes, 'names': names, 'shared_name': shared_name, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    capacity = 2
    dtypes = [np.string_]
    shapes = [[None, None]]
    names = [None] * 1 if [None] else None
    shared_name = None
    name = 'padding_fifo_queue_3'
    input_dict = {'capacity': capacity, 'dtypes': dtypes, 'shapes': shapes, 'names': names, 'shared_name': shared_name, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    capacity = 7
    dtypes = [np.bool_]
    shapes = [[None, 5, 5]]
    names = [None] * 1 if [None] else None
    shared_name = 'shared_queue_4'
    name = 'padding_fifo_queue_4'
    input_dict = {'capacity': capacity, 'dtypes': dtypes, 'shapes': shapes, 'names': names, 'shared_name': shared_name, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    capacity = 3
    dtypes = [np.complex64]
    shapes = [[None]]
    names = [None] * 1 if [None] else None
    shared_name = None
    name = 'padding_fifo_queue_5'
    input_dict = {'capacity': capacity, 'dtypes': dtypes, 'shapes': shapes, 'names': names, 'shared_name': shared_name, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    capacity = 6
    dtypes = [np.int64, np.float16]
    shapes = [[None, 1], [2, None, 4]]
    names = ['big_int', 'small_float']
    shared_name = 'shared_queue_6'
    name = 'padding_fifo_queue_6'
    input_dict = {'capacity': capacity, 'dtypes': dtypes, 'shapes': shapes, 'names': names, 'shared_name': shared_name, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    capacity = 4
    dtypes = [np.uint8]
    shapes = [[None, 28, 28, 3]]
    names = [None] * 1 if [None] else None
    shared_name = None
    name = 'padding_fifo_queue_7'
    input_dict = {'capacity': capacity, 'dtypes': dtypes, 'shapes': shapes, 'names': names, 'shared_name': shared_name, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    capacity = 8
    dtypes = [np.int8]
    shapes = [[None, None, None]]
    names = [None] * 1 if [None] else None
    shared_name = 'shared_queue_8'
    name = 'padding_fifo_queue_8'
    input_dict = {'capacity': capacity, 'dtypes': dtypes, 'shapes': shapes, 'names': names, 'shared_name': shared_name, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    capacity = 9
    dtypes = [np.float32, np.int32, np.string_]
    shapes = [[None], [None, 5], [1, None, 2]]
    names = ['float_values', 'int_matrix', 'string_array']
    shared_name = None
    name = 'padding_fifo_queue_9'
    input_dict = {'capacity': capacity, 'dtypes': dtypes, 'shapes': shapes, 'names': names, 'shared_name': shared_name, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    capacity = 1
    dtypes = [np.uint16]
    shapes = [[None, 16, 16]]
    names = [None] * 1 if [None] else None
    shared_name = 'shared_queue_10'
    name = 'padding_fifo_queue_10'
    input_dict = {'capacity': capacity, 'dtypes': dtypes, 'shapes': shapes, 'names': names, 'shared_name': shared_name, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.queue.PaddingFIFOQueue"] = tf_queue_paddingFIFOQueue_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.queue.PaddingFIFOQueue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.queue.PaddingFIFOQueue'.")

check_valid('tf.queue.PaddingFIFOQueue', generated_inputs['tf.queue.PaddingFIFOQueue'], lib="tf", suffix=0)
