
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf

def tf_queue_priorityqueue_inputs():
    """
    Generates a list of valid inputs for the tf.queue.PriorityQueue function.
    """
    list_of_inputs = []

    # Input 1: Basic case with two components
    input_dict_1 = {
        'capacity': 10,
        'types': ['int64', 'float32'],
        'shapes': [[], [10]],
        'names': ['priority', 'data'],
        'shared_name': 'q1',
        'name': 'basic_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Multiple data components with different types
    input_dict_2 = {
        'capacity': 100,
        'types': ['int64', 'float64', 'string'],
        'shapes': [[], [3, 4], []],
        'names': ['p', 'd', 's'],
        'shared_name': 'q2',
        'name': 'multi_component_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Optional shapes and names are not provided
    input_dict_3 = {
        'capacity': 20,
        'types': ['int64', 'complex64'],
        'shapes': [[], [2, 2]],
        'names': ['priority', 'complex_data'],
        'shared_name': 'my_shared_queue',
        'name': 'shared_queue_instance'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Shapes with empty dimensions (representing scalars)
    input_dict_4 = {
        'capacity': 15,
        'types': ['int64', 'float32', 'int32'],
        'shapes': [[], [], []],
        'names': ['priority', 'value1', 'value2'],
        'shared_name': 'q4',
        'name': 'scalar_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: High capacity queue
    input_dict_5 = {
        'capacity': 10000,
        'types': ['int64', 'float16'],
        'shapes': [[], [128, 128]],
        'names': ['priority', 'image'],
        'shared_name': 'q5',
        'name': 'large_capacity_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: All arguments provided
    input_dict_6 = {
        'capacity': 42,
        'types': ['int64', 'float64'],
        'shapes': [[], [3, 3, 3]],
        'names': ['p_level', 'data_cube'],
        'shared_name': 'another_shared_queue',
        'name': 'fully_specified_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Single component (only priority)
    input_dict_7 = {
        'capacity': 8,
        'types': ['int64'],
        'shapes': [[]],
        'names': ['priority_only'],
        'shared_name': 'q7',
        'name': 'priority_only_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    
    # Input 8: Shapes with zero dimension
    input_dict_8 = {
        'capacity': 5,
        'types': ['int64', 'float32'],
        'shapes': [[], [0, 10]],
        'names': ['priority', 'empty_data'],
        'shared_name': 'q8',
        'name': 'zero_dim_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Using bfloat16
    input_dict_9 = {
        'capacity': 12,
        'types': ['int64', 'bfloat16'],
        'shapes': [[], [64, 64]],
        'names': ['p', 'bf16_data'],
        'shared_name': 'bfloat_q',
        'name': 'bfloat_q_instance'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Minimal capacity
    input_dict_10 = {
        'capacity': 1,
        'types': ['int64', 'uint8'],
        'shapes': [[], []],
        'names': ['p', 'data'],
        'shared_name': 'q10',
        'name': 'minimal_capacity_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))
    
    # Input 11: Using an empty string for shared_name
    input_dict_11 = {
        'capacity': 10,
        'types': ['int64', 'bool'],
        'shapes': [[], []],
        'names': ['priority', 'flag'],
        'shared_name': '',
        'name': 'empty_shared_name_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

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
