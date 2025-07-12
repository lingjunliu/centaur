
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
        "types": [np.float32],
        "shapes": [(1,)],
        "names": ['float_val'],
        "shared_name": "queue1",
        "name": "priority_queue_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "capacity": 10,
        "types": [np.int32, np.float32],
        "shapes": [(), ()],
        "names": ['int_val', 'float_val'],
        "shared_name": "queue2",
        "name": "priority_queue_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "capacity": 20,
        "types": [np.int64, np.float64, np.bool_],
        "shapes": [(1, 2), (3,), ()],
        "names": ['int64_val', 'float64_val', 'bool_val'],
        "shared_name": "queue3",
        "name": "priority_queue_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "capacity": 1,
        "types": [tf.string],
        "shapes": [()],
        "names": ['string_val'],
        "shared_name": "queue4",
        "name": "priority_queue_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "capacity": 7,
        "types": [np.complex64],
        "shapes": [(2, 2)],
        "names": ['complex_val'],
        "shared_name": "queue5",
        "name": "priority_queue_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "capacity": 15,
        "types": [np.uint8, np.int16],
        "shapes": [(1,), (2,1,3)],
        "names": ['uint8_val', 'int16_val'],
        "shared_name": "queue6",
        "name": "priority_queue_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "capacity": 3,
        "types": [np.float16],
        "shapes": [()],
        "names": ['float16_val'],
        "shared_name": "queue7",
        "name": "priority_queue_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "capacity": 12,
        "types": [tf.dtypes.as_dtype(np.object_)],
        "shapes": [(1,3,5)],
        "names": ['object_val'],
        "shared_name": "queue8",
        "name": "priority_queue_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input_dict = {
        "capacity": 8,
        "types": [np.int8],
        "shapes": [(4,4)],
        "names": ['int8_val'],
        "shared_name": "queue9",
        "name": "priority_queue_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "capacity": 2,
        "types": [np.uint32, np.uint64],
        "shapes": [(), (1, 1)],
        "names": ['uint32_val', 'uint64_val'],
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
