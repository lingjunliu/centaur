
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_RandomShuffleQueue_inputs():
    list_of_inputs = []

    # Input 1
    component_types = [tf.float32]
    shapes = [[2, 3]]
    capacity = 10
    min_after_dequeue = 3
    seed = 123
    seed2 = 456
    container = "container1"
    shared_name = "shared_queue1"
    name = "queue_op1"

    input_dict = {
        "component_types": component_types,
        "shapes": shapes,
        "capacity": capacity,
        "min_after_dequeue": min_after_dequeue,
        "seed": seed,
        "seed2": seed2,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    component_types = [tf.int32]
    shapes = [[5]]
    capacity = 5
    min_after_dequeue = 1
    seed = 789
    seed2 = 101
    container = "container2"
    shared_name = "shared_queue2"
    name = "queue_op2"

    input_dict = {
        "component_types": component_types,
        "shapes": shapes,
        "capacity": capacity,
        "min_after_dequeue": min_after_dequeue,
        "seed": seed,
        "seed2": seed2,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    component_types = [tf.bool]
    shapes = []
    capacity = -1
    min_after_dequeue = 0
    seed = 0
    seed2 = 0
    container = ""
    shared_name = ""
    name = "queue_op3"

    input_dict = {
        "component_types": component_types,
        "shapes": shapes,
        "capacity": capacity,
        "min_after_dequeue": min_after_dequeue,
        "seed": seed,
        "seed2": seed2,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    component_types = [tf.float64]
    shapes = [[1, 10]]
    capacity = 20
    min_after_dequeue = 5
    seed = -1
    seed2 = -2
    container = "container4"
    shared_name = "shared_queue4"
    name = "queue_op4"

    input_dict = {
        "component_types": component_types,
        "shapes": shapes,
        "capacity": capacity,
        "min_after_dequeue": min_after_dequeue,
        "seed": seed,
        "seed2": seed2,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    component_types = [tf.uint8]
    shapes = [[4, 4, 4, 4]]
    capacity = 1
    min_after_dequeue = 0
    seed = 1
    seed2 = 2
    container = "container5"
    shared_name = "shared_queue5"
    name = "queue_op5"

    input_dict = {
        "component_types": component_types,
        "shapes": shapes,
        "capacity": capacity,
        "min_after_dequeue": min_after_dequeue,
        "seed": seed,
        "seed2": seed2,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    component_types = [tf.qint8]
    shapes = [[]]
    capacity = 100
    min_after_dequeue = 20
    seed = 1000
    seed2 = 2000
    container = ""
    shared_name = "shared_queue6"
    name = "queue_op6"

    input_dict = {
        "component_types": component_types,
        "shapes": shapes,
        "capacity": capacity,
        "min_after_dequeue": min_after_dequeue,
        "seed": seed,
        "seed2": seed2,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    component_types = [tf.float16]
    shapes = []
    capacity = 50
    min_after_dequeue = 10
    seed = 555
    seed2 = 666
    container = "bfloat_container"
    shared_name = "bfloat_queue"
    name = "queue_op9"
    
    input_dict = {
        "component_types": component_types,
        "shapes": shapes,
        "capacity": capacity,
        "min_after_dequeue": min_after_dequeue,
        "seed": seed,
        "seed2": seed2,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    component_types = [tf.string]
    shapes = []
    capacity = 50
    min_after_dequeue = 10
    seed = 555
    seed2 = 666
    container = ""
    shared_name = ""
    name = "queue_op10"
    
    input_dict = {
        "component_types": component_types,
        "shapes": shapes,
        "capacity": capacity,
        "min_after_dequeue": min_after_dequeue,
        "seed": seed,
        "seed2": seed2,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9
    component_types = [tf.complex64]
    shapes = [[3,3,3]]
    capacity = 20
    min_after_dequeue = 5
    seed = -1
    seed2 = -2
    container = "container4"
    shared_name = "shared_queue4"
    name = "queue_op4"

    input_dict = {
        "component_types": component_types,
        "shapes": shapes,
        "capacity": capacity,
        "min_after_dequeue": min_after_dequeue,
        "seed": seed,
        "seed2": seed2,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    component_types = [tf.bfloat16]
    shapes = [[1,2,3,4,5]]
    capacity = 50
    min_after_dequeue = 10
    seed = 555
    seed2 = 666
    container = "bfloat_container"
    shared_name = "bfloat_queue"
    name = "queue_op8"
    
    input_dict = {
        "component_types": component_types,
        "shapes": shapes,
        "capacity": capacity,
        "min_after_dequeue": min_after_dequeue,
        "seed": seed,
        "seed2": seed2,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.RandomShuffleQueue"] = tf_raw_ops_RandomShuffleQueue_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.RandomShuffleQueue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RandomShuffleQueue'.")

check_valid('tf.raw_ops.RandomShuffleQueue', generated_inputs['tf.raw_ops.RandomShuffleQueue'], lib="tf", suffix=0)
