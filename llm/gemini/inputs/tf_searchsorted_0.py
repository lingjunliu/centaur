
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_searchsorted_inputs():
    list_of_inputs = []

    # Input 1
    sorted_sequence = tf.constant([1, 3, 5, 7, 9], dtype=tf.float32).numpy()
    values = tf.constant([2, 4, 6, 8, 10], dtype=tf.float32).numpy()
    side = "left"
    out_type = tf.int32
    name = None
    input_dict = {"sorted_sequence": sorted_sequence, "values": values, "side": side, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    sorted_sequence = tf.constant([1, 3, 5, 7, 9], dtype=tf.float32).numpy()
    values = tf.constant([2, 4, 6, 8, 10], dtype=tf.float32).numpy()
    side = "right"
    out_type = tf.int64
    name = "search"
    input_dict = {"sorted_sequence": sorted_sequence, "values": values, "side": side, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    sorted_sequence = tf.constant([1, 3, 5, 7, 9], dtype=tf.float32).numpy()
    values = tf.constant([1, 3, 5, 7, 9], dtype=tf.float32).numpy()
    side = "left"
    out_type = tf.int32
    name = None
    input_dict = {"sorted_sequence": sorted_sequence, "values": values, "side": side, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    sorted_sequence = tf.constant([1, 3, 5, 7, 9], dtype=tf.float32).numpy()
    values = tf.constant([1, 3, 5, 7, 9], dtype=tf.float32).numpy()
    side = "right"
    out_type = tf.int64
    name = "search"
    input_dict = {"sorted_sequence": sorted_sequence, "values": values, "side": side, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    sorted_sequence = tf.constant([[1, 3, 5], [7, 9, 11]], dtype=tf.float32).numpy()
    values = tf.constant([[2, 4], [8, 10]], dtype=tf.float32).numpy()
    side = "left"
    out_type = tf.int32
    name = None
    input_dict = {"sorted_sequence": sorted_sequence, "values": values, "side": side, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    sorted_sequence = tf.constant([[1, 3, 5], [7, 9, 11]], dtype=tf.float32).numpy()
    values = tf.constant([[2, 4], [8, 10]], dtype=tf.float32).numpy()
    side = "right"
    out_type = tf.int64
    name = "search"
    input_dict = {"sorted_sequence": sorted_sequence, "values": values, "side": side, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    sorted_sequence = tf.constant([1, 3, 5, 7, 9], dtype=tf.int32).numpy()
    values = tf.constant([2, 4, 6, 8, 10], dtype=tf.int32).numpy()
    side = "left"
    out_type = tf.int32
    name = None
    input_dict = {"sorted_sequence": sorted_sequence, "values": values, "side": side, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    sorted_sequence = tf.constant([1, 3, 5, 7, 9], dtype=tf.int32).numpy()
    values = tf.constant([2, 4, 6, 8, 10], dtype=tf.int32).numpy()
    side = "right"
    out_type = tf.int64
    name = "search"
    input_dict = {"sorted_sequence": sorted_sequence, "values": values, "side": side, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    sorted_sequence = tf.constant([-5, -3, -1, 1, 3], dtype=tf.float32).numpy()
    values = tf.constant([-4, -2, 0, 2, 4], dtype=tf.float32).numpy()
    side = "left"
    out_type = tf.int32
    name = None
    input_dict = {"sorted_sequence": sorted_sequence, "values": values, "side": side, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    sorted_sequence = tf.constant([-5, -3, -1, 1, 3], dtype=tf.float32).numpy()
    values = tf.constant([-4, -2, 0, 2, 4], dtype=tf.float32).numpy()
    side = "right"
    out_type = tf.int64
    name = "search"
    input_dict = {"sorted_sequence": sorted_sequence, "values": values, "side": side, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11 - empty values
    sorted_sequence = tf.constant([1, 3, 5, 7, 9], dtype=tf.float32).numpy()
    values = np.array([], dtype=np.float32)
    side = "left"
    out_type = tf.int32
    name = None
    input_dict = {"sorted_sequence": sorted_sequence, "values": values, "side": side, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12 - empty sorted_sequence
    sorted_sequence = np.array([], dtype=np.float32)
    values = tf.constant([1, 2, 3], dtype=tf.float32).numpy()
    side = "left"
    out_type = tf.int32
    name = None
    input_dict = {"sorted_sequence": sorted_sequence, "values": values, "side": side, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.searchsorted"] = tf_searchsorted_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.searchsorted' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.searchsorted'.")

check_valid('tf.searchsorted', generated_inputs['tf.searchsorted'], lib="tf", suffix=0)
