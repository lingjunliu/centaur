
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_PaddingFIFOQueue_inputs():
    list_of_inputs = []

    # Input 1
    component_types = [tf.float32.as_numpy_dtype]
    shapes = [[10]]
    capacity = 10
    container = ""
    shared_name = ""
    name = "queue1"
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
    component_types = [tf.int32.as_numpy_dtype, tf.float64.as_numpy_dtype]
    shapes = [[5], [2, 2]]
    capacity = 5
    container = "container2"
    shared_name = "shared2"
    name = "queue2"
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
    shapes = [[1, 10]]
    capacity = -1
    container = "container3"
    shared_name = "shared3"
    name = "queue3"
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
    shapes = []
    capacity = 100
    container = ""
    shared_name = ""
    name = "queue4"
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
    component_types = [tf.uint8.as_numpy_dtype, tf.int16.as_numpy_dtype, tf.float16.as_numpy_dtype]
    shapes = [[], [2], [1, 2, 3]]
    capacity = 20
    container = "container5"
    shared_name = "shared5"
    name = "queue5"
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
    shapes = [[-1]]
    capacity = 30
    container = ""
    shared_name = ""
    name = "queue6"
    input_dict = {
        "component_types": component_types,
        "shapes": shapes,
        "capacity": capacity,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 - Removed tf.resource.as_numpy_dtype as it's causing issues
    # Input 8 - Removed tf.variant.as_numpy_dtype as it's causing issues
    # Input 9 - Removed tf.qint8.as_numpy_dtype and tf.quint8.as_numpy_dtype as they are causing issues.

   # Input 10
    component_types = [tf.double.as_numpy_dtype]
    shapes = [[2, 0, 3]]
    capacity = 50
    container = "container10"
    shared_name = "shared10"
    name = "queue10"
    input_dict = {
        "component_types": component_types,
        "shapes": shapes,
        "capacity": capacity,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    #Input 11
    component_types = [tf.int32.as_numpy_dtype]
    shapes = [[1, 2, 3]]
    capacity = 50
    container = ""
    shared_name = ""
    name = "queue11"

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
generated_inputs["tf.raw_ops.PaddingFIFOQueue"] = tf_raw_ops_PaddingFIFOQueue_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.PaddingFIFOQueue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.PaddingFIFOQueue'.")

check_valid('tf.raw_ops.PaddingFIFOQueue', generated_inputs['tf.raw_ops.PaddingFIFOQueue'], lib="tf", suffix=0)
