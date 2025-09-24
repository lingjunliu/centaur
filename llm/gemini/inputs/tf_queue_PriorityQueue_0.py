
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_queue_priorityqueue_inputs():
    """
    Generates a list of valid inputs for tf.queue.PriorityQueue.
    """
    list_of_inputs = []

    # Input 1: Basic two-component queue
    input_dict_1 = {
        'capacity': 10,
        'types': ['int64', 'float32'],
        'shapes': None,
        'names': None,
        'shared_name': None,
        'name': 'q1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Three components, with names
    input_dict_2 = {
        'capacity': 20,
        'types': ['int64', 'string', 'bool'],
        'shapes': None,
        'names': ['prio', 'meta', 'flag'],
        'shared_name': None,
        'name': 'q2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Using a shared_name
    input_dict_3 = {
        'capacity': 5,
        'types': ['int64', 'int32'],
        'shapes': None,
        'names': None,
        'shared_name': 'shared_q3',
        'name': 'q3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Priority only queue
    input_dict_4 = {
        'capacity': 100,
        'types': ['int64'],
        'shapes': None,
        'names': None,
        'shared_name': None,
        'name': 'q4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Another data type (uint8)
    input_dict_5 = {
        'capacity': 50,
        'types': ['int64', 'uint8'],
        'shapes': None,
        'names': None,
        'shared_name': None,
        'name': 'q5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: All optional arguments specified
    input_dict_6 = {
        'capacity': 25,
        'types': ['int64', 'complex64'],
        'shapes': None,
        'names': ['priority', 'complex_data'],
        'shared_name': 'shared_q6',
        'name': 'q6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Minimal capacity and empty shared_name
    input_dict_7 = {
        'capacity': 1,
        'types': ['int64', 'float16'],
        'shapes': None,
        'names': None,
        'shared_name': '',
        'name': 'q7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Minimal arguments with just priority
    input_dict_8 = {
        'capacity': 2,
        'types': ['int64'],
        'shapes': None,
        'names': None,
        'shared_name': None,
        'name': 'q10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Input 9: More components
    input_dict_9 = {
        'capacity': 12,
        'types': ['int64', 'int16', 'string'],
        'shapes': None,
        'names': ['id', 'value', 'desc'],
        'shared_name': 'high_dim_q',
        'name': 'q_high_dim'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Using quantized type
    input_dict_10 = {
        'capacity': 40,
        'types': ['int64', 'qint8'],
        'shapes': None,
        'names': None,
        'shared_name': None,
        'name': 'quantized_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.queue.PriorityQueue"] = tf_queue_priorityqueue_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.queue.PriorityQueue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.queue.PriorityQueue'.")

check_valid('tf.queue.PriorityQueue', generated_inputs['tf.queue.PriorityQueue'], lib="tf", suffix=0)
