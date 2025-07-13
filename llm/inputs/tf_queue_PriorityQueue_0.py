
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_queue_priority_queue_inputs():
    list_of_inputs = []

    # Input 1
    capacity = 5
    types = [tf.int32]
    shapes = [[]]
    names = ['priority']
    shared_name = 'queue1'
    name = 'priority_queue_1'
    input_dict = {
        'capacity': capacity,
        'types': types,
        'shapes': shapes,
        'names': names,
        'shared_name': shared_name,
        'name': name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    capacity = 10
    types = [tf.float32, tf.int64]
    shapes = [[], []]
    names = ['priority', 'data']
    shared_name = 'queue2'
    name = 'priority_queue_2'
    input_dict = {
        'capacity': capacity,
        'types': types,
        'shapes': shapes,
        'names': names,
        'shared_name': shared_name,
        'name': name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    capacity = 2
    types = [tf.string, tf.float64, tf.int32]
    shapes = [[], [], []]
    names = ['priority', 'feature1', 'feature2']
    shared_name = 'queue3'
    name = 'priority_queue_3'
    input_dict = {
        'capacity': capacity,
        'types': types,
        'shapes': shapes,
        'names': names,
        'shared_name': shared_name,
        'name': name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    capacity = 7
    types = [tf.int32, tf.float32]
    shapes = [[2], [3]]
    names = ['priority', 'vector']
    shared_name = 'queue4'
    name = 'priority_queue_4'
    input_dict = {
        'capacity': capacity,
        'types': types,
        'shapes': shapes,
        'names': names,
        'shared_name': shared_name,
        'name': name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5
    capacity = 3
    types = [tf.int32, tf.float32, tf.string]
    shapes = [[], [2, 2], []]
    names = ['priority', 'matrix', 'message']
    shared_name = 'queue5'
    name = 'priority_queue_5'
    input_dict = {
        'capacity': capacity,
        'types': types,
        'shapes': shapes,
        'names': names,
        'shared_name': shared_name,
        'name': name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    capacity = 1
    types = [tf.int32]
    shapes = [[1, 1, 1]]
    names = ['priority']
    shared_name = 'queue6'
    name = 'priority_queue_6'
    input_dict = {
        'capacity': capacity,
        'types': types,
        'shapes': shapes,
        'names': names,
        'shared_name': shared_name,
        'name': name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    capacity = 4
    types = [tf.int64, tf.float64]
    shapes = [[], [4, 4]]
    names = ['priority', 'data']
    shared_name = 'queue7'
    name = 'priority_queue_7'
    input_dict = {
        'capacity': capacity,
        'types': types,
        'shapes': shapes,
        'names': names,
        'shared_name': shared_name,
        'name': name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    capacity = 6
    types = [tf.string, tf.int32]
    shapes = [[], []]
    names = ['priority', 'count']
    shared_name = 'queue8'
    name = 'priority_queue_8'
    input_dict = {
        'capacity': capacity,
        'types': types,
        'shapes': shapes,
        'names': names,
        'shared_name': shared_name,
        'name': name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    capacity = 8
    types = [tf.int32]
    shapes = [[]]
    names = ['priority']
    shared_name = ''
    name = 'priority_queue_9'
    input_dict = {
        'capacity': capacity,
        'types': types,
        'shapes': shapes,
        'names': names,
        'shared_name': shared_name,
        'name': name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    capacity = 9
    types = [tf.int32, tf.float32]
    shapes = [[], []]
    names = ['priority', 'value']
    shared_name = 'queue10'
    name = ''
    input_dict = {
        'capacity': capacity,
        'types': types,
        'shapes': shapes,
        'names': names,
        'shared_name': shared_name,
        'name': name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11 - Added back since it might be required to have a shape to at least pass the shape check
    capacity = 2
    types = [tf.int32]
    shapes = [[]]
    names = ["priority"]
    shared_name = "queue11"
    name = "priority_queue_11"
    input_dict = {
        'capacity': capacity,
        'types': types,
        'shapes': shapes,
        'names': names,
        'shared_name': shared_name,
        'name': name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.queue.PriorityQueue"] = tf_queue_priority_queue_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.queue.PriorityQueue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.queue.PriorityQueue'.")

check_valid('tf.queue.PriorityQueue', generated_inputs['tf.queue.PriorityQueue'], lib="tf", suffix=0)
