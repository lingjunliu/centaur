
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_raw_ops_FIFOQueue_inputs():
    list_of_inputs = []

    # Input 1
    component_types = [tf.float32]
    shapes = []
    capacity = -1
    container = ""
    shared_name = ""
    name = None
    input_dict = {"component_types": component_types, "shapes": shapes, "capacity": capacity, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    component_types = [tf.int32]
    shapes = [[10]]
    capacity = 100
    container = "my_container"
    shared_name = "my_shared_queue"
    name = "my_queue"
    input_dict = {"component_types": component_types, "shapes": shapes, "capacity": capacity, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    component_types = [tf.bool]
    shapes = [[2, 2]]
    capacity = 5
    container = ""
    shared_name = "another_shared_queue"
    name = None
    input_dict = {"component_types": component_types, "shapes": shapes, "capacity": capacity, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    component_types = [tf.float64]
    shapes = [[5, 5]]
    capacity = -1
    container = "complex_container"
    shared_name = ""
    name = "complex_queue"
    input_dict = {"component_types": component_types, "shapes": shapes, "capacity": capacity, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    component_types = [tf.uint8]
    shapes = []
    capacity = 1
    container = ""
    shared_name = ""
    name = None
    input_dict = {"component_types": component_types, "shapes": shapes, "capacity": capacity, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    component_types = [tf.qint8]
    shapes = [[3, 4]]
    capacity = 20
    container = "quantized_container"
    shared_name = "quantized_queue"
    name = None
    input_dict = {"component_types": component_types, "shapes": shapes, "capacity": capacity, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7
    component_types = [tf.float16]
    shapes = []
    capacity = -1
    container = ""
    shared_name = "float16_shared_name"
    name = "float16_queue"
    input_dict = {"component_types": component_types, "shapes": shapes, "capacity": capacity, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    component_types = [tf.string]
    shapes = []
    capacity = 15
    container = "string_container"
    shared_name = ""
    name = "string_queue"
    input_dict = {"component_types": component_types, "shapes": shapes, "capacity": capacity, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    component_types = [tf.bfloat16]
    shapes = []
    capacity = -1
    container = ""
    shared_name = ""
    name = None
    input_dict = {"component_types": component_types, "shapes": shapes, "capacity": capacity, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.FIFOQueue"] = tf_raw_ops_FIFOQueue_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.FIFOQueue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FIFOQueue'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.FIFOQueue', generated_inputs['tf.raw_ops.FIFOQueue'], lib="tf", suffix=0)
