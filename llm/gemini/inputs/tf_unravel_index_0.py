
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_unravel_index_inputs():
    list_of_inputs = []

    # Input 1
    indices = np.array([2, 5, 7], dtype=np.int32)
    dims = np.array([3, 3], dtype=np.int32)
    name = "unravel_1"
    input_dict = {"indices": indices, "dims": dims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    indices = np.array(7, dtype=np.int64)
    dims = np.array([3, 3], dtype=np.int64)
    name = "unravel_2"
    input_dict = {"indices": indices, "dims": dims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    indices = np.array([2, 5, 7, 10, 12], dtype=np.int32)
    dims = np.array([4, 4], dtype=np.int32)
    name = "unravel_3"
    input_dict = {"indices": indices, "dims": dims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    indices = np.array([15], dtype=np.int64)
    dims = np.array([5, 3, 2], dtype=np.int64)
    name = "unravel_4"
    input_dict = {"indices": indices, "dims": dims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    indices = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    dims = np.array([5], dtype=np.int32)
    name = "unravel_5"
    input_dict = {"indices": indices, "dims": dims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    indices = np.array([0], dtype=np.int64)
    dims = np.array([1, 1, 1], dtype=np.int64)
    name = "unravel_6"
    input_dict = {"indices": indices, "dims": dims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    indices = np.array([23], dtype=np.int32)
    dims = np.array([2, 3, 4], dtype=np.int32)
    name = "unravel_7"
    input_dict = {"indices": indices, "dims": dims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    indices = np.array([1, 3, 5, 7, 9], dtype=np.int64)
    dims = np.array([2, 5], dtype=np.int64)
    name = "unravel_8"
    input_dict = {"indices": indices, "dims": dims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    indices = np.array([0, 4, 8, 12, 16], dtype=np.int32)
    dims = np.array([4, 5], dtype=np.int32)
    name = "unravel_9"
    input_dict = {"indices": indices, "dims": dims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    indices = np.array([119], dtype=np.int64)
    dims = np.array([10, 6, 2], dtype=np.int64)
    name = "unravel_10"
    input_dict = {"indices": indices, "dims": dims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    indices = np.array([0, 1, 2, 3], dtype=np.int32)
    dims = np.array([2,2], dtype=np.int32)
    name = "unravel_11"
    input_dict = {"indices": indices, "dims": dims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    indices = np.array([0], dtype=np.int64)
    dims = np.array([1], dtype=np.int64)
    name = "unravel_12"
    input_dict = {"indices": indices, "dims": dims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.unravel_index"] = tf_unravel_index_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.unravel_index' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.unravel_index'.")

check_valid('tf.unravel_index', generated_inputs['tf.unravel_index'], lib="tf", suffix=0)
