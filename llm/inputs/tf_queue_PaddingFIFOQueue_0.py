
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_queue_padding_fifo_queue_inputs():
    list_of_inputs = []

    # Input 1
    capacity = 5
    dtypes = [np.float32]
    shapes = [[2, None]]
    names = None
    shared_name = None
    name = "queue1"
    input_dict = {"capacity": capacity, "dtypes": dtypes, "shapes": shapes, "names": names, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    capacity = 10
    dtypes = [np.int32, np.float64]
    shapes = [[None], [3, 3]]
    names = ["id", "matrix"]
    shared_name = "shared_queue"
    name = "queue2"
    input_dict = {"capacity": capacity, "dtypes": dtypes, "shapes": shapes, "names": names, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    capacity = 3
    dtypes = [np.string_]
    shapes = [[None, None]]
    names = None
    shared_name = None
    name = "queue3"
    input_dict = {"capacity": capacity, "dtypes": dtypes, "shapes": shapes, "names": names, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    capacity = 7
    dtypes = [np.bool_]
    shapes = [[1, 5, None]]
    names = None
    shared_name = "another_shared_queue"
    name = "queue4"
    input_dict = {"capacity": capacity, "dtypes": dtypes, "shapes": shapes, "names": names, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    capacity = 12
    dtypes = [np.complex64]
    shapes = [[None, 8]]
    names = ["complex_data"]
    shared_name = None
    name = "queue5"
    input_dict = {"capacity": capacity, "dtypes": dtypes, "shapes": shapes, "names": names, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    capacity = 2
    dtypes = [np.uint8]
    shapes = [[4, None, 2]]
    names = None
    shared_name = "yet_another_shared_queue"
    name = "queue6"
    input_dict = {"capacity": capacity, "dtypes": dtypes, "shapes": shapes, "names": names, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    capacity = 8
    dtypes = [np.int64, np.float16]
    shapes = [[None, 10], [2]]
    names = ["long_data", "short_float"]
    shared_name = None
    name = "queue7"
    input_dict = {"capacity": capacity, "dtypes": dtypes, "shapes": shapes, "names": names, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    capacity = 4
    dtypes = [np.float32, np.int32, np.string_]
    shapes = [[None], [1,1], []]
    names = ["a", "b", "c"]
    shared_name = "shared_name_42"
    name = "queue8"
    input_dict = {"capacity": capacity, "dtypes": dtypes, "shapes": shapes, "names": names, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    capacity = 15
    dtypes = [np.float64]
    shapes = [[None, None, None]]
    names = None
    shared_name = None
    name = "queue9"
    input_dict = {"capacity": capacity, "dtypes": dtypes, "shapes": shapes, "names": names, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    capacity = 1
    dtypes = [np.int8]
    shapes = [[None]]
    names = ["tiny_int"]
    shared_name = "queue_one_shared"
    name = "queue10"
    input_dict = {"capacity": capacity, "dtypes": dtypes, "shapes": shapes, "names": names, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    capacity = 20
    dtypes = [np.uint32]
    shapes = [[1]]
    names = None
    shared_name = None
    name = "queue11"
    input_dict = {"capacity": capacity, "dtypes": dtypes, "shapes": shapes, "names": names, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12
    capacity = 6
    dtypes = [np.float32, np.int32]
    shapes = [[2, None], [None]]
    names = ["feature", "label"]
    shared_name = "queue12_shared"
    name = "queue12"
    input_dict = {"capacity": capacity, "dtypes": dtypes, "shapes": shapes, "names": names, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 13
    capacity = 9
    dtypes = [np.complex128]
    shapes = [[None, None]]
    names = ["complex_data_2d"]
    shared_name = None
    name = "queue13"
    input_dict = {"capacity": capacity, "dtypes": dtypes, "shapes": shapes, "names": names, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 14
    capacity = 11
    dtypes = [np.float32, np.int32, np.string_]
    shapes = [[None, 5], [3], [1, None]]
    names = ["features", "labels", "metadata"]
    shared_name = "queue14_shared"
    name = "queue14"
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
