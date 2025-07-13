
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_random_shuffle_queue_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "component_types": [tf.float32.as_numpy_dtype],
        "shapes": [],
        "capacity": 10,
        "min_after_dequeue": 2,
        "seed": 1,
        "seed2": 2,
        "container": "",
        "shared_name": "",
        "name": "queue1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "component_types": [tf.int32.as_numpy_dtype, tf.float64.as_numpy_dtype],
        "shapes": [[10], []],
        "capacity": 5,
        "min_after_dequeue": 1,
        "seed": 0,
        "seed2": 0,
        "container": "container2",
        "shared_name": "shared_queue2",
        "name": "queue2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "component_types": [tf.string.as_numpy_dtype],
        "shapes": [[5, 5]],
        "capacity": -1,
        "min_after_dequeue": 5,
        "seed": 123,
        "seed2": 456,
        "container": "",
        "shared_name": "shared_queue3",
        "name": "queue3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "component_types": [tf.bool.as_numpy_dtype, tf.complex64.as_numpy_dtype, tf.uint8.as_numpy_dtype],
        "shapes": [[], [2, 2], [3]],
        "capacity": 20,
        "min_after_dequeue": 0,
        "seed": 789,
        "seed2": 101,
        "container": "container4",
        "shared_name": "",
        "name": "queue4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5
    input_dict = {
        "component_types": [tf.int64.as_numpy_dtype],
        "shapes": [[2, 3, 4]],
        "capacity": 1,
        "min_after_dequeue": 1,
        "seed": -1,
        "seed2": -2,
        "container": "",
        "shared_name": "shared_queue5",
        "name": "queue5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "component_types": [tf.float16.as_numpy_dtype],
        "shapes": [[]],
        "capacity": 100,
        "min_after_dequeue": 50,
        "seed": 0,
        "seed2": 1000,
        "container": "container6",
        "shared_name": "",
        "name": "queue6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "component_types": [tf.qint8.as_numpy_dtype, tf.quint8.as_numpy_dtype],
        "shapes": [[1], [1]],
        "capacity": -1,
        "min_after_dequeue": -1,
        "seed": 2000,
        "seed2": 0,
        "container": "",
        "shared_name": "shared_queue7",
        "name": "queue7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: single component_type, shape
    input_dict = {
        "component_types": [tf.float64.as_numpy_dtype],
        "shapes": [[3, 2]],
        "capacity": 15,
        "min_after_dequeue": 8,
        "seed": 1234,
        "seed2": 4321,
        "container": "container8",
        "shared_name": "shared_queue8",
        "name": "queue8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: min_after_dequeue close to capacity
    input_dict = {
        "component_types": [tf.int32.as_numpy_dtype],
        "shapes": [],
        "capacity": 10,
        "min_after_dequeue": 8,
        "seed": 5678,
        "seed2": 8765,
        "container": "container9",
        "shared_name": "shared_queue9",
        "name": "queue9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: empty shared_name
    input_dict = {
        "component_types": [tf.string.as_numpy_dtype],
        "shapes": [[1, 1]],
        "capacity": 5,
        "min_after_dequeue": 2,
        "seed": 9012,
        "seed2": 2109,
        "container": "container10",
        "shared_name": "",
        "name": "queue10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.RandomShuffleQueue"] = tf_raw_ops_random_shuffle_queue_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.RandomShuffleQueue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RandomShuffleQueue'.")

check_valid('tf.raw_ops.RandomShuffleQueue', generated_inputs['tf.raw_ops.RandomShuffleQueue'], lib="tf", suffix=0)
