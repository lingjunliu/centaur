
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
    shapes = [(10,)]
    names = ['input1']
    shared_name = 'queue1'
    name = 'fifo_queue1'
    input_dict = {
        "capacity": capacity,
        "dtypes": dtypes,
        "shapes": list(shapes),
        "names": names,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    capacity = 10
    dtypes = [tf.float32, tf.int64]
    shapes = [(), ()]
    names = ['input2', 'input3']
    shared_name = 'queue2'
    name = 'fifo_queue2'
    input_dict = {
        "capacity": capacity,
        "dtypes": dtypes,
        "shapes": list(shapes),
        "names": names,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    capacity = 2
    dtypes = [tf.string]
    shapes = [(2, 2)]
    names = ['input4']
    shared_name = 'queue3'
    name = 'fifo_queue3'
    input_dict = {
        "capacity": capacity,
        "dtypes": dtypes,
        "shapes": list(shapes),
        "names": names,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    capacity = 7
    dtypes = [tf.bool]
    shapes = [()]
    names = ['input5']
    shared_name = 'queue4'
    name = 'fifo_queue4'
    input_dict = {
        "capacity": capacity,
        "dtypes": dtypes,
        "shapes": list(shapes),
        "names": names,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    capacity = 3
    dtypes = [tf.int32, tf.float64, tf.string]
    shapes = [(1,), (), (2, 3)]
    names = ['input6', 'input7', 'input8']
    shared_name = 'queue5'
    name = 'fifo_queue5'
    input_dict = {
        "capacity": capacity,
        "dtypes": dtypes,
        "shapes": list(shapes),
        "names": names,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    capacity = 1
    dtypes = [tf.float16]
    shapes = [(5, 5)]
    names = ['input9']
    shared_name = 'queue6'
    name = 'fifo_queue6'
    input_dict = {
        "capacity": capacity,
        "dtypes": dtypes,
        "shapes": list(shapes),
        "names": names,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    capacity = 8
    dtypes = [tf.complex64]
    shapes = [()]
    names = ['input10']
    shared_name = 'queue7'
    name = 'fifo_queue7'
    input_dict = {
        "capacity": capacity,
        "dtypes": dtypes,
        "shapes": list(shapes),
        "names": names,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    capacity = 4
    dtypes = [tf.resource]
    shapes = [()]
    names = ['input11']
    shared_name = 'queue8'
    name = 'fifo_queue8'
    input_dict = {
        "capacity": capacity,
        "dtypes": dtypes,
        "shapes": list(shapes),
        "names": names,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    capacity = 6
    dtypes = [tf.variant]
    shapes = [()]
    names = ['input12']
    shared_name = 'queue9'
    name = 'fifo_queue9'
    input_dict = {
        "capacity": capacity,
        "dtypes": dtypes,
        "shapes": list(shapes),
        "names": names,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    capacity = 9
    dtypes = [tf.uint8]
    shapes = [(1, 2, 3)]
    names = ['input13']
    shared_name = 'queue10'
    name = 'fifo_queue10'
    input_dict = {
        "capacity": capacity,
        "dtypes": dtypes,
        "shapes": list(shapes),
        "names": names,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.queue.FIFOQueue"] = tf_queue_fifoqueue_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.queue.FIFOQueue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.queue.FIFOQueue'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.queue.FIFOQueue', generated_inputs['tf.queue.FIFOQueue'], lib="tf", suffix=0)
