
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_barrier_inputs():
    list_of_inputs = []

    # Input 1
    component_types = [tf.float32.as_numpy_dtype]
    shapes = []
    capacity = -1
    container = ""
    shared_name = ""
    name = "barrier_1"

    input_dict = {
        "component_types": component_types,
        "shapes": shapes,
        "capacity": capacity,
        "container": container,
        "shared_name": shared_name,
        "name": name,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    component_types = [tf.int32.as_numpy_dtype, tf.float64.as_numpy_dtype]
    shapes = [[1], [1]]
    capacity = 10
    container = "test_container"
    shared_name = "test_shared_name"
    name = "barrier_2"

    input_dict = {
        "component_types": component_types,
        "shapes": shapes,
        "capacity": capacity,
        "container": container,
        "shared_name": shared_name,
        "name": name,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    component_types = [tf.string.as_numpy_dtype]
    shapes = []
    capacity = 100
    container = ""
    shared_name = "barrier_shared"
    name = None

    input_dict = {
        "component_types": component_types,
        "shapes": shapes,
        "capacity": capacity,
        "container": container,
        "shared_name": shared_name,
        "name": name,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    component_types = [tf.bool.as_numpy_dtype]
    shapes = [[1]]
    capacity = -1
    container = "another_container"
    shared_name = ""
    name = "barrier_4"

    input_dict = {
        "component_types": component_types,
        "shapes": shapes,
        "capacity": capacity,
        "container": container,
        "shared_name": shared_name,
        "name": name,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 5
    component_types = [tf.uint8.as_numpy_dtype, tf.int16.as_numpy_dtype]
    shapes = [[1], [1]]
    capacity = 5
    container = ""
    shared_name = ""
    name = "barrier_5"

    input_dict = {
        "component_types": component_types,
        "shapes": shapes,
        "capacity": capacity,
        "container": container,
        "shared_name": shared_name,
        "name": name,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    component_types = [tf.int64.as_numpy_dtype]
    shapes = []
    capacity = 20
    container = "container_six"
    shared_name = "shared_six"
    name = None

    input_dict = {
        "component_types": component_types,
        "shapes": shapes,
        "capacity": capacity,
        "container": container,
        "shared_name": shared_name,
        "name": name,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Barrier"] = tf_raw_ops_barrier_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Barrier' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Barrier'.")

check_valid('tf.raw_ops.Barrier', generated_inputs['tf.raw_ops.Barrier'], lib="tf", suffix=0)
