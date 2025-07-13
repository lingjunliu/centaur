
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
    shapes = [[2, 2]]
    capacity = 10
    container = "test_container_1"
    shared_name = "test_shared_name_1"
    name = "test_queue_1"

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
    shapes = [[3], [2, 2]]
    capacity = 5
    container = "test_container_2"
    shared_name = "test_shared_name_2"
    name = "test_queue_2"

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
    shapes = [[None]]
    capacity = -1
    container = ""
    shared_name = ""
    name = "test_queue_3"

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
    shapes = [[2, None, 3]]
    capacity = 100
    container = "bool_container"
    shared_name = "bool_shared"
    name = "bool_queue"

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
    component_types = [tf.complex64.as_numpy_dtype]
    shapes = []
    capacity = 20
    container = "complex_container"
    shared_name = "complex_shared"
    name = "complex_queue"

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
    component_types = [tf.int64.as_numpy_dtype, tf.float16.as_numpy_dtype, tf.string.as_numpy_dtype]
    shapes = [[], [1, 5], [3, 2]]
    capacity = 7
    container = "multi_container"
    shared_name = "multi_shared"
    name = "multi_queue"

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
    component_types = [tf.qint8.as_numpy_dtype]
    shapes = [[4, 4]]
    capacity = 12
    container = "qint_container"
    shared_name = "qint_shared"
    name = "qint_queue"

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
    component_types = [tf.uint8.as_numpy_dtype]
    shapes = [[10]]
    capacity = 10
    container = "uint8_container"
    shared_name = "uint8_shared"
    name = "uint8_queue"

    input_dict = {
        "component_types": component_types,
        "shapes": shapes,
        "capacity": capacity,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    component_types = [tf.float32.as_numpy_dtype, tf.int32.as_numpy_dtype]
    shapes = [[None], []]
    capacity = 5
    container = "mixed_container"
    shared_name = "mixed_shared"
    name = "mixed_queue"
    input_dict = {
        "component_types": component_types,
        "shapes": shapes,
        "capacity": capacity,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    component_types = [tf.float32.as_numpy_dtype]
    shapes = []
    capacity = -1
    container = ""
    shared_name = ""
    name = "queue_no_shape"

    input_dict = {
        "component_types": component_types,
        "shapes": shapes,
        "capacity": capacity,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Remove the shape.
    component_types = [tf.float64.as_numpy_dtype]
    shapes = []
    capacity = 10
    container = "var_shape_container"
    shared_name = "var_shape_shared"
    name = "var_shape_queue"
    input_dict = {
        "component_types": component_types,
        "shapes": shapes,
        "capacity": capacity,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    component_types = [tf.int32.as_numpy_dtype]
    shapes = [[None, None]]
    capacity = 15
    container = "dynamic_container"
    shared_name = "dynamic_shared"
    name = "dynamic_queue"
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
