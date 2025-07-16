
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_queue_priority_queue_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "capacity": 5,
        "types": [tf.float32],
        "shapes": [(1,)],
        "names": ['float_priority'],
        "shared_name": "queue1",
        "name": "priority_queue_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "capacity": 10,
        "types": [tf.int32, tf.float32],
        "shapes": [(2,), ()],
        "names": ['int_priority', 'float_data'],
        "shared_name": "queue2",
        "name": "priority_queue_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "capacity": 2,
        "types": [tf.string, tf.int64],
        "shapes": [(), ()],
        "names": ['string_priority', 'int64_data'],
        "shared_name": "queue3",
        "name": "priority_queue_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "capacity": 7,
        "types": [tf.bool, tf.float64, tf.int32],
        "shapes": [(), (3, 3), ()],
        "names": ['bool_priority', 'float64_data', 'int32_data'],
        "shared_name": "queue4",
        "name": "priority_queue_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "capacity": 3,
        "types": [tf.complex64],
        "shapes": [(2, 2)],
        "names": ['complex_priority'],
        "shared_name": "queue5",
        "name": "priority_queue_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6
    input_dict = {
        "capacity": 15,
        "types": [tf.uint8],
        "shapes": [(1,)],
        "names": ['uint8_priority'],
        "shared_name": "queue6",
        "name": "priority_queue_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "capacity": 20,
        "types": [tf.int8, tf.uint16],
        "shapes": [(), ()],
        "names": ['int8_priority', 'uint16_data'],
        "shared_name": "queue7",
        "name": "priority_queue_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "capacity": 1,
        "types": [tf.string],
        "shapes": [()],
        "names": ['string_priority'],
        "shared_name": "queue8",
        "name": "priority_queue_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "capacity": 4,
        "types": [tf.float64],
        "shapes": [(2,)],
        "names": ['float64_priority'],
        "shared_name": "queue9",
        "name": "priority_queue_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "capacity": 8,
        "types": [tf.bfloat16],
        "shapes": [(4,4)],
        "names": ['bfloat16_priority'],
        "shared_name": "queue10",
        "name": "priority_queue_10"
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
