
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy


def tf_raw_ops_priority_queue_inputs():
    list_of_inputs = []

    # Input 1: Basic case, single float component, unlimited capacity.
    input_dict_1 = {
        'component_types': [np.float32],
        'shapes': [[10]],
        'capacity': -1,
        'container': '',
        'shared_name': '',
        'name': 'queue1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Multiple components, fixed capacity.
    input_dict_2 = {
        'component_types': [np.int32, np.float64],
        'shapes': [[1], [5, 2]],
        'capacity': 100,
        'container': '',
        'shared_name': '',
        'name': 'queue2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Unconstrained shapes.
    input_dict_3 = {
        'component_types': [np.string_],
        'shapes': [],
        'capacity': 50,
        'container': '',
        'shared_name': '',
        'name': 'queue3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: No extra components (priority only).
    input_dict_4 = {
        'component_types': [],
        'shapes': [],
        'capacity': 20,
        'container': '',
        'shared_name': '',
        'name': 'queue4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Shared queue with a container.
    input_dict_5 = {
        'component_types': [np.bool_],
        'shapes': [[1]],
        'capacity': -1,
        'container': 'my_container',
        'shared_name': 'shared_queue5',
        'name': 'queue5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Zero capacity queue.
    input_dict_6 = {
        'component_types': [np.uint8],
        'shapes': [[1]],
        'capacity': 0,
        'container': '',
        'shared_name': '',
        'name': 'queue6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: High-dimensional tensor component.
    input_dict_7 = {
        'component_types': [np.int16],
        'shapes': [[2, 3, 4]],
        'capacity': 1000,
        'container': '',
        'shared_name': '',
        'name': 'queue7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Multiple scalar components.
    input_dict_8 = {
        'component_types': [np.float16, np.int64],
        'shapes': [[1], [1]],
        'capacity': 5,
        'container': '',
        'shared_name': '',
        'name': 'queue8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Unconstrained shapes with multiple component types.
    input_dict_9 = {
        'component_types': [np.float32, np.int32, np.string_],
        'shapes': [],
        'capacity': 10,
        'container': 'another_container',
        'shared_name': '',
        'name': 'queue9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Just a name.
    input_dict_10 = {
        'component_types': [],
        'shapes': [],
        'capacity': -1,
        'container': '',
        'shared_name': '',
        'name': 'a_very_specific_queue_name'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.PriorityQueue"] = tf_raw_ops_priority_queue_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.PriorityQueue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.PriorityQueue'.")

check_valid('tf.raw_ops.PriorityQueue', generated_inputs['tf.raw_ops.PriorityQueue'], lib="tf", suffix=0)
