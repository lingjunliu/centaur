
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_fifo_queue_inputs():
    list_of_inputs = []

    # Input 1
    component_types = [tf.float32]
    shapes = []
    capacity = -1
    container = ""
    shared_name = ""
    name = "fifo_queue_1"
    input_dict = {
        "component_types": component_types,
        "shapes": shapes,
        "capacity": capacity,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    component_types = [tf.int32, tf.float64]
    shapes = [[10], []]
    capacity = 100
    container = "my_container"
    shared_name = "shared_queue_2"
    name = "fifo_queue_2"
    input_dict = {
        "component_types": component_types,
        "shapes": shapes,
        "capacity": capacity,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    component_types = [tf.string]
    shapes = [[5, 5]]
    capacity = 50
    container = "another_container"
    shared_name = "shared_queue_3"
    name = None
    input_dict = {
        "component_types": component_types,
        "shapes": shapes,
        "capacity": capacity,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    component_types = [tf.bool]
    shapes = []
    capacity = 1
    container = ""
    shared_name = ""
    name = "fifo_queue_4"
    input_dict = {
        "component_types": component_types,
        "shapes": shapes,
        "capacity": capacity,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 5
    component_types = [tf.float32, tf.int32, tf.string]
    shapes = [[10, 10], [5], []]
    capacity = 20
    container = "container_5"
    shared_name = "shared_5"
    name = "fifo_queue_5"
    input_dict = {
        "component_types": component_types,
        "shapes": shapes,
        "capacity": capacity,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.FIFOQueue"] = tf_raw_ops_fifo_queue_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.FIFOQueue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FIFOQueue'.")

check_valid('tf.raw_ops.FIFOQueue', generated_inputs['tf.raw_ops.FIFOQueue'], lib="tf", suffix=0)
