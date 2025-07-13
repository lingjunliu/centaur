
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
    shapes = [[1, 2]]
    capacity = 10
    container = "container1"
    shared_name = "barrier1"
    name = "op1"

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
    component_types = [tf.int32.as_numpy_dtype]
    shapes = [[1, 3]]
    capacity = -1
    container = ""
    shared_name = ""
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

    # Input 3
    component_types = [tf.string.as_numpy_dtype]
    shapes = [[1]]
    capacity = 100
    container = "container2"
    shared_name = "barrier2"
    name = "op3"

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
    component_types = [tf.bool.as_numpy_dtype]
    shapes = [[1, 1, 1]]
    capacity = 5
    container = ""
    shared_name = "barrier4"
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

    # Input 5
    component_types = [tf.uint8.as_numpy_dtype]
    shapes = [[1, 1]]
    capacity = 20
    container = "container5"
    shared_name = ""
    name = "op5"

    input_dict = {
        "component_types": component_types,
        "shapes": shapes,
        "capacity": capacity,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    component_types = [tf.complex64.as_numpy_dtype]
    shapes = [[1, 7, 7, 7]]
    capacity = 12
    container = ""
    shared_name = "barrier6"
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

   # Input 7
    component_types = [tf.int8.as_numpy_dtype]
    shapes = [[1]]
    capacity = 30
    container = "container7"
    shared_name = ""
    name = "op7"

    input_dict = {
        "component_types": component_types,
        "shapes": shapes,
        "capacity": capacity,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    component_types = [tf.float32.as_numpy_dtype]
    shapes = [[1, 8]]
    capacity = 15
    container = ""
    shared_name = "barrier8"
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

    # Input 9
    component_types = [tf.qint8.as_numpy_dtype]
    shapes = [[1, 10, 10]]
    capacity = 8
    container = "container9"
    shared_name = "barrier9"
    name = "op9"

    input_dict = {
        "component_types": component_types,
        "shapes": shapes,
        "capacity": capacity,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    component_types = [tf.quint16.as_numpy_dtype]
    shapes = [[1, 4, 4, 4]]
    capacity = 40
    container = ""
    shared_name = ""
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
