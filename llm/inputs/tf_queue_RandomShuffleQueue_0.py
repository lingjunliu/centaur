
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_queue_randomshufflequeue_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        'capacity': 10,
        'min_after_dequeue': 5,
        'dtypes': [np.int32],
        'shapes': [(2,)],
        'names': ['data'],
        'seed': 123,
        'shared_name': 'queue1',
        'name': 'random_queue_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'capacity': 5,
        'min_after_dequeue': 2,
        'dtypes': [np.float32, np.int64],
        'shapes': [(), ()],
        'names': ['float_data', 'int_data'],
        'seed': None,
        'shared_name': 'queue2',
        'name': 'random_queue_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'capacity': 20,
        'min_after_dequeue': 10,
        'dtypes': [tf.string],
        'shapes': [(3, 3)],
        'names': [str('string_data')],
        'seed': 456,
        'shared_name': 'queue3',
        'name': 'random_queue_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'capacity': 8,
        'min_after_dequeue': 3,
        'dtypes': [np.bool_],
        'shapes': [(1, 4, 2)],
        'names': ['bool_data'],
        'seed': None,
        'shared_name': 'queue4',
        'name': 'random_queue_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'capacity': 12,
        'min_after_dequeue': 6,
        'dtypes': [np.int32, np.float64, tf.string],
        'shapes': [(), (2, 1), (1,)],
        'names': ['int_data', 'float_data', 'string_data'],
        'seed': 789,
        'shared_name': 'queue5',
        'name': 'random_queue_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'capacity': 30,
        'min_after_dequeue': 15,
        'dtypes': [np.int8],
        'shapes': [(5,)],
        'names': ['data'],
        'seed': 101,
        'shared_name': 'queue6',
        'name': 'random_queue_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7
    input_dict = {
        'capacity': 7,
        'min_after_dequeue': 1,
        'dtypes': [np.uint8],
        'shapes': [(7, 7)],
        'names': ['data'],
        'seed': 112,
        'shared_name': 'queue7',
        'name': 'random_queue_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'capacity': 9,
        'min_after_dequeue': 4,
        'dtypes': [np.int16, np.float16],
        'shapes': [(), ()],
        'names': ['int_data', 'float_data'],
        'seed': 1234,
        'shared_name': 'queue8',
        'name': 'random_queue_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'capacity': 15,
        'min_after_dequeue': 7,
        'dtypes': [np.int64],
        'shapes': [(4, 4, 4)],
        'names': ['data'],
        'seed': 5678,
        'shared_name': 'queue9',
        'name': 'random_queue_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'capacity': 40,
        'min_after_dequeue': 20,
        'dtypes': [np.float32, np.int32, np.bool_],
        'shapes': [(1,), (3,), ()],
        'names': ['float_data', 'int_data', 'bool_data'],
        'seed': 9012,
        'shared_name': 'queue10',
        'name': 'random_queue_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.queue.RandomShuffleQueue"] = tf_queue_randomshufflequeue_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.queue.RandomShuffleQueue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.queue.RandomShuffleQueue'.")

check_valid('tf.queue.RandomShuffleQueue', generated_inputs['tf.queue.RandomShuffleQueue'], lib="tf", suffix=0)
