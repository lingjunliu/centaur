
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
    capacity = 10
    container = ""
    shared_name = ""
    name = ""
    input_dict = {
        "component_types": component_types,
        "shapes": shapes,
        "capacity": capacity,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(input_dict)

    # Input 2
    component_types = [tf.int32, tf.float64]
    shapes = [[10], [5, 5]]
    capacity = -1
    container = "test_container"
    shared_name = "shared_queue_2"
    name = ""
    input_dict = {
        "component_types": component_types,
        "shapes": shapes,
        "capacity": capacity,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(input_dict)

    # Input 3
    component_types = [tf.string]
    shapes = []
    capacity = 0
    container = ""
    shared_name = ""
    name = ""
    input_dict = {
        "component_types": component_types,
        "shapes": shapes,
        "capacity": capacity,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(input_dict)

    # Input 4
    component_types = [tf.bool, tf.int64, tf.complex64]
    shapes = [[], [2, 3], [1, 1, 1]]
    capacity = 100
    container = "another_container"
    shared_name = "shared_queue_4"
    name = ""
    input_dict = {
        "component_types": component_types,
        "shapes": shapes,
        "capacity": capacity,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(input_dict)

    # Input 5
    component_types = [tf.float16]
    shapes = [[5, 5, 5, 5]]
    capacity = 1
    container = ""
    shared_name = ""
    name = ""
    input_dict = {
        "component_types": component_types,
        "shapes": shapes,
        "capacity": capacity,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(input_dict)

    # Input 6
    component_types = [tf.uint8]
    shapes = []
    capacity = -1
    container = "container_6"
    shared_name = "shared_6"
    name = ""
    input_dict = {
        "component_types": component_types,
        "shapes": shapes,
        "capacity": capacity,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(input_dict)

    # Input 7
    component_types = [tf.qint8, tf.quint8]
    shapes = [[2, 2], [3]]
    capacity = 20
    container = ""
    shared_name = "shared_7"
    name = ""
    input_dict = {
        "component_types": component_types,
        "shapes": shapes,
        "capacity": capacity,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(input_dict)

    # Input 8
    component_types = [tf.resource]
    shapes = []
    capacity = 5
    container = "res_container"
    shared_name = ""
    name = ""
    input_dict = {
        "component_types": component_types,
        "shapes": shapes,
        "capacity": capacity,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(input_dict)

    # Input 9
    component_types = [tf.variant]
    shapes = [[1,1,1]]
    capacity = 15
    container = ""
    shared_name = "var_queue"
    name = ""
    input_dict = {
        "component_types": component_types,
        "shapes": shapes,
        "capacity": capacity,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(input_dict)

    # Input 10
    component_types = [tf.float32, tf.int32, tf.string]
    shapes = [[10, 10], [5], []]
    capacity = -1
    container = "all_types_container"
    shared_name = "shared_all_types"
    name = ""
    input_dict = {
        "component_types": component_types,
        "shapes": shapes,
        "capacity": capacity,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(input_dict)

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
