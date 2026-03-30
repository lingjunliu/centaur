
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_queue_priority_queue_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "capacity": 10,
        "types": [tf.int32, tf.float32],
        "shapes": [(), (1,)],
        "names": ["priority", "data"],
        "shared_name": "queue1",
        "name": "priority_queue1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "capacity": 5,
        "types": [tf.float64],
        "shapes": [(2, 2)],
        "names": ["priority"],
        "shared_name": "queue2",
        "name": "priority_queue2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "capacity": 20,
        "types": [tf.int64, tf.bool],
        "shapes": [(), ()],
        "names": ["priority", "data"],
        "shared_name": "queue3",
        "name": "priority_queue3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "capacity": 1,
        "types": [tf.uint8],
        "shapes": [()],
        "names": ["priority"],
        "shared_name": "queue4",
        "name": "priority_queue4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "capacity": 100,
        "types": [tf.int32],
        "shapes": [()],
        "names": ["priority"],
        "shared_name": "queue5",
        "name": "priority_queue5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "capacity": 2,
        "types": [tf.float32],
        "shapes": [()],
        "names": ["priority"],
        "shared_name": "queue6",
        "name": "priority_queue6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "capacity": 7,
        "types": [tf.int32],
        "shapes": [()],
        "names": ["priority"],
        "shared_name": "queue7",
        "name": "priority_queue7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "capacity": 3,
        "types": [tf.float32],
        "shapes": [()],
        "names": ["priority"],
        "shared_name": "queue8",
        "name": "priority_queue8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9
    input_dict = {
        "capacity": 12,
        "types": [tf.int32],
        "shapes": [()],
        "names": ["priority"],
        "shared_name": "queue9",
        "name": "priority_queue9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "capacity": 4,
        "types": [tf.float64],
        "shapes": [()],
        "names": ["priority"],
        "shared_name": "queue10",
        "name": "priority_queue10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.queue.PriorityQueue"] = tf_queue_priority_queue_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.queue.PriorityQueue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.queue.PriorityQueue'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.queue.PriorityQueue', generated_inputs['tf.queue.PriorityQueue'], lib="tf", suffix=0)
