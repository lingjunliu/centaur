
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
    name = None
    input_dict = {"component_types": component_types, "shapes": shapes, "capacity": capacity, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    component_types = [tf.int32.as_numpy_dtype, tf.float64.as_numpy_dtype]
    shapes = [[5], [2, 3]]
    capacity = 5
    container = "my_container"
    shared_name = "my_shared_queue"
    name = "my_queue"
    input_dict = {"component_types": component_types, "shapes": shapes, "capacity": capacity, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    component_types = [tf.string.as_numpy_dtype]
    shapes = []
    capacity = -1
    container = ""
    shared_name = ""
    name = None
    input_dict = {"component_types": component_types, "shapes": shapes, "capacity": capacity, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    component_types = [tf.bool.as_numpy_dtype]
    shapes = [[], []]
    capacity = 100
    container = "bool_container"
    shared_name = "bool_queue"
    name = "bool_name"
    input_dict = {"component_types": component_types, "shapes": shapes, "capacity": capacity, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    component_types = [tf.float32.as_numpy_dtype, tf.int64.as_numpy_dtype]
    shapes = [[None], [2, None]]
    capacity = 20
    container = "mixed_container"
    shared_name = "mixed_queue"
    name = "mixed_name"
    input_dict = {"component_types": component_types, "shapes": shapes, "capacity": capacity, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    component_types = [tf.complex64.as_numpy_dtype]
    shapes = [[2, 2, 2]]
    capacity = 10
    container = ""
    shared_name = ""
    name = None
    input_dict = {"component_types": component_types, "shapes": shapes, "capacity": capacity, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    component_types = [tf.uint8.as_numpy_dtype]
    shapes = [[3, 4]]
    capacity = -1
    container = "uint8_container"
    shared_name = "uint8_queue"
    name = "uint8_name"
    input_dict = {"component_types": component_types, "shapes": shapes, "capacity": capacity, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    component_types = [tf.float16.as_numpy_dtype]
    shapes = [[1, 5, 10]]
    capacity = 15
    container = ""
    shared_name = "float16_queue"
    name = None
    input_dict = {"component_types": component_types, "shapes": shapes, "capacity": capacity, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.PaddingFIFOQueue"] = tf_raw_ops_PaddingFIFOQueue_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.PaddingFIFOQueue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.PaddingFIFOQueue'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.PaddingFIFOQueue', generated_inputs['tf.raw_ops.PaddingFIFOQueue'], lib="tf", suffix=0)
