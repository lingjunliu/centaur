
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy


def tf_queue_priorityqueue_inputs():
    list_of_inputs = []

    # Corresponds to tf.DType enums:
    # float32: 1, float64: 2, int32: 3, uint8: 4, int16: 5, int8: 6,
    # string: 7, complex64: 8, int64: 9, bool: 10, float16: 19

    # Input 1: Basic case with two types and scalar shapes
    input_dict = {
        'capacity': 10,
        'types': [9, 1],
        'shapes': [[], []],
        'names': [],
        'shared_name': '',
        'name': 'basic_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Defined shapes for a scalar priority and a 1D vector
    input_dict = {
        'capacity': 100,
        'types': [9, 7],
        'shapes': [(), (10,)],
        'names': [],
        'shared_name': '',
        'name': 'vector_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multiple components with different shapes
    input_dict = {
        'capacity': 50,
        'types': [9, 3, 10],
        'shapes': [(), (3, 4), ()],
        'names': [],
        'shared_name': '',
        'name': 'multi_component_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Using component names
    input_dict = {
        'capacity': 20,
        'types': [9, 2],
        'shapes': [(), ()],
        'names': ['priority', 'value'],
        'shared_name': '',
        'name': 'named_components_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Using a shared name for inter-process sharing
    input_dict = {
        'capacity': 1000,
        'types': [9, 4],
        'shapes': [(), (64, 64)],
        'names': [],
        'shared_name': 'my_shared_image_queue',
        'name': 'image_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Single component queue (priority only)
    input_dict = {
        'capacity': 30,
        'types': [9],
        'shapes': [()],
        'names': [],
        'shared_name': '',
        'name': 'priority_only_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Partially defined shape (e.g., for variable-sized batches)
    input_dict = {
        'capacity': 5,
        'types': [9, 1],
        'shapes': [(), (None, 224, 224, 3)],
        'names': [],
        'shared_name': '',
        'name': 'variable_batch_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: All optional parameters are specified
    input_dict = {
        'capacity': 15,
        'types': [9, 5, 8],
        'shapes': [(), (5,), (2, 2)],
        'names': ['prio', 'data', 'c_matrix'],
        'shared_name': 'fully_specified_shared',
        'name': 'fully_specified_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large capacity queue
    input_dict = {
        'capacity': 10000,
        'types': [9, 6],
        'shapes': [(), (1024,)],
        'names': [],
        'shared_name': '',
        'name': 'large_capacity_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Shapes list with None elements, indicating unconstrained shapes for components
    input_dict = {
        'capacity': 25,
        'types': [9, 19, 7],
        'shapes': [None, None, None],
        'names': ['priority', 'embedding', 'metadata'],
        'shared_name': '',
        'name': 'unconstrained_shape_list_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

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
