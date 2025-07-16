
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_queue_fifoqueue_inputs():
    list_of_inputs = []

    # Input 1
    capacity = 5
    dtypes = [tf.int32]
    shapes = [None]
    names = None
    shared_name = None
    name = "fifo_queue_1"
    input_dict = {"capacity": capacity, "dtypes": dtypes, "shapes": shapes, "names": names, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    capacity = 10
    dtypes = [tf.float32, tf.int64]
    shapes = [None, (2, 2)]
    names = ["float_data", "int_matrix"]
    shared_name = "shared_queue_2"
    name = "fifo_queue_2"
    input_dict = {"capacity": capacity, "dtypes": dtypes, "shapes": shapes, "names": names, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    capacity = 3
    dtypes = [tf.string]
    shapes = [None]
    names = None
    shared_name = "shared_queue_3"
    name = "fifo_queue_3"
    input_dict = {"capacity": capacity, "dtypes": dtypes, "shapes": shapes, "names": names, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    capacity = 7
    dtypes = [tf.bool, tf.int32, tf.float64]
    shapes = [None, (3,), (2, 1, 4)]
    names = ["flag", "ids", "data"]
    shared_name = None
    name = "fifo_queue_4"
    input_dict = {"capacity": capacity, "dtypes": dtypes, "shapes": shapes, "names": names, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    capacity = 2
    dtypes = [tf.uint8]
    shapes = [(100, 100, 3)]
    names = None
    shared_name = "shared_queue_5"
    name = "fifo_queue_5"
    input_dict = {"capacity": capacity, "dtypes": dtypes, "shapes": shapes, "names": names, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    capacity = 4
    dtypes = [tf.int16]
    shapes = [None]
    names = None
    shared_name = None
    name = "fifo_queue_6"
    input_dict = {"capacity": capacity, "dtypes": dtypes, "shapes": shapes, "names": names, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    capacity = 6
    dtypes = [tf.float16]
    shapes = [(5, 5)]
    names = None
    shared_name = "shared_queue_7"
    name = "fifo_queue_7"
    input_dict = {"capacity": capacity, "dtypes": dtypes, "shapes": shapes, "names": names, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    capacity = 8
    dtypes = [tf.complex64]
    shapes = [(2,)]
    names = None
    shared_name = None
    name = "fifo_queue_8"
    input_dict = {"capacity": capacity, "dtypes": dtypes, "shapes": shapes, "names": names, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    capacity = 9
    dtypes = [tf.resource]
    shapes = [None]
    names = None
    shared_name = "shared_queue_9"
    name = "fifo_queue_9"
    input_dict = {"capacity": capacity, "dtypes": dtypes, "shapes": shapes, "names": names, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    capacity = 1
    dtypes = [tf.variant]
    shapes = [None]
    names = None
    shared_name = None
    name = "fifo_queue_10"
    input_dict = {"capacity": capacity, "dtypes": dtypes, "shapes": shapes, "names": names, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.queue.FIFOQueue"] = tf_queue_fifoqueue_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.queue.FIFOQueue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.queue.FIFOQueue'.")

check_valid('tf.queue.FIFOQueue', generated_inputs['tf.queue.FIFOQueue'], lib="tf", suffix=0)
