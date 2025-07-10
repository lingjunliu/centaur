
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_queue_padding_fifo_queue_inputs():
    list_of_inputs = []

    # Input 1
    capacity = 10
    dtypes = [np.float32]
    shapes = [[None]]
    names = None
    shared_name = ""
    name = "queue1"
    input_dict = {"capacity": capacity, "dtypes": dtypes, "shapes": shapes, "names": names, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    capacity = 5
    dtypes = [np.int32, np.float64]
    shapes = [[None], [2, None]]
    names = ["int_data", "float_data"]
    shared_name = "shared_queue_2"
    name = "queue2"
    input_dict = {"capacity": capacity, "dtypes": dtypes, "shapes": shapes, "names": names, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    capacity = 20
    dtypes = [np.bytes_, np.bool_]
    shapes = [[None, 3], [None]]
    names = None
    shared_name = "shared_queue_3"
    name = "queue3"
    input_dict = {"capacity": capacity, "dtypes": dtypes, "shapes": shapes, "names": names, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    capacity = 1
    dtypes = [np.complex64]
    shapes = [[5, None, 2]]
    names = ["complex_data"]
    shared_name = ""
    name = "queue4"
    input_dict = {"capacity": capacity, "dtypes": dtypes, "shapes": shapes, "names": names, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    capacity = 15
    dtypes = [np.uint8, np.int64]
    shapes = [[None], [None, 10]]
    names = ["uint_data", "int64_data"]
    shared_name = "shared_queue_5"
    name = "queue5"
    input_dict = {"capacity": capacity, "dtypes": dtypes, "shapes": shapes, "names": names, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    capacity = 7
    dtypes = [np.float16]
    shapes = [[None, None]]
    names = None
    shared_name = ""
    name = "queue6"
    input_dict = {"capacity": capacity, "dtypes": dtypes, "shapes": shapes, "names": names, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7
    capacity = 3
    dtypes = [np.int16]
    shapes = [[1,None,5]]
    names = ["int16_data"]
    shared_name = "shared_queue_7"
    name = "queue7"
    input_dict = {"capacity": capacity, "dtypes": dtypes, "shapes": shapes, "names": names, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    capacity = 9
    dtypes = [np.complex128, np.uint32]
    shapes = [[None], [None, None, 3]]
    names = None
    shared_name = ""
    name = "queue8"
    input_dict = {"capacity": capacity, "dtypes": dtypes, "shapes": shapes, "names": names, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    capacity = 4
    dtypes = [np.bool_]
    shapes = [[None, 4, None]]
    names = ["bool_data"]
    shared_name = "shared_queue_9"
    name = "queue9"
    input_dict = {"capacity": capacity, "dtypes": dtypes, "shapes": shapes, "names": names, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    capacity = 12
    dtypes = [np.int8]
    shapes = [[None, None, None]]
    names = None
    shared_name = ""
    name = "queue10"
    input_dict = {"capacity": capacity, "dtypes": dtypes, "shapes": shapes, "names": names, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.queue.PaddingFIFOQueue"] = tf_queue_padding_fifo_queue_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.queue.PaddingFIFOQueue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.queue.PaddingFIFOQueue'.")

check_valid('tf.queue.PaddingFIFOQueue', generated_inputs['tf.queue.PaddingFIFOQueue'], lib="tf", suffix=0)
