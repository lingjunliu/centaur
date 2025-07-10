
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
    dtypes = [np.int32]
    shapes = [(2, 2)]
    names = ['input1']
    shared_name = 'queue1'
    name = 'fifo_queue1'

    input_dict = {
        "capacity": capacity,
        "dtypes": dtypes,
        "shapes": shapes,
        "names": [str(x) for x in names],
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    capacity = 10
    dtypes = [np.float32, np.int64]
    shapes = [(), ()]
    names = ['input2_float', 'input2_int']
    shared_name = 'queue2'
    name = 'fifo_queue2'

    input_dict = {
        "capacity": capacity,
        "dtypes": dtypes,
        "shapes": shapes,
        "names": [str(x) for x in names],
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    capacity = 3
    dtypes = [tf.string]
    shapes = [(1, 5)]
    names = ['input3']
    shared_name = 'queue3'
    name = 'fifo_queue3'

    input_dict = {
        "capacity": capacity,
        "dtypes": dtypes,
        "shapes": shapes,
        "names": [str(x) for x in names],
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    capacity = 7
    dtypes = [np.bool_]
    shapes = [(3, 1, 4)]
    names = ['input4']
    shared_name = 'queue4'
    name = 'fifo_queue4'

    input_dict = {
        "capacity": capacity,
        "dtypes": dtypes,
        "shapes": shapes,
        "names": [str(x) for x in names],
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5
    capacity = 1
    dtypes = [np.int32, np.float64, tf.string]
    shapes = [(1,), (), (2, 1)]
    names = ['input5_int', 'input5_float', 'input5_string']
    shared_name = 'queue5'
    name = 'fifo_queue5'

    input_dict = {
        "capacity": capacity,
        "dtypes": dtypes,
        "shapes": shapes,
        "names": [str(x) for x in names],
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    capacity = 20
    dtypes = [np.uint8]
    shapes = [(4,4,4)]
    names = ['input6']
    shared_name = 'queue6'
    name = 'fifo_queue6'

    input_dict = {
        "capacity": capacity,
        "dtypes": dtypes,
        "shapes": shapes,
        "names": [str(x) for x in names],
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    capacity = 4
    dtypes = [np.int16]
    shapes = [None]
    names = ['input7']
    shared_name = 'queue7'
    name = 'fifo_queue7'

    input_dict = {
        "capacity": capacity,
        "dtypes": dtypes,
        "shapes": shapes,
        "names": [str(x) for x in names],
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    capacity = 6
    dtypes = [np.float16, np.int8]
    shapes = [(2,2), (3,)]
    names = ['input8_float', 'input8_int']
    shared_name = 'queue8'
    name = 'fifo_queue8'

    input_dict = {
        "capacity": capacity,
        "dtypes": dtypes,
        "shapes": shapes,
        "names": [str(x) for x in names],
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    capacity = 12
    dtypes = [np.complex64]
    shapes = [(5,)]
    names = ['input9']
    shared_name = 'queue9'
    name = 'fifo_queue9'

    input_dict = {
        "capacity": capacity,
        "dtypes": dtypes,
        "shapes": shapes,
        "names": [str(x) for x in names],
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    capacity = 8
    dtypes = [np.uint32, np.int64]
    shapes = [(), ()]
    names = ['input10_uint', 'input10_int']
    shared_name = 'queue10'
    name = 'fifo_queue10'

    input_dict = {
        "capacity": capacity,
        "dtypes": dtypes,
        "shapes": shapes,
        "names": [str(x) for x in names],
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
    
    print("Valid")

if 'tf.queue.FIFOQueue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.queue.FIFOQueue'.")

check_valid('tf.queue.FIFOQueue', generated_inputs['tf.queue.FIFOQueue'], lib="tf", suffix=0)
