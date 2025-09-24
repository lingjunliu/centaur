
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_argsort_inputs():
    list_of_inputs = []

    # Input 1
    values = tf.constant([1, 10, 26.9, 2.8, 166.32, 62.3], dtype=tf.float32).numpy()
    axis = -1
    direction = 'ASCENDING'
    stable = False
    name = None

    input_dict = {
        "values": values,
        "axis": axis,
        "direction": direction,
        "stable": stable,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    values = tf.constant([[30,20,10],[20,10,30],[10,30,20]], dtype=tf.int32).numpy()
    axis = 0
    direction = 'DESCENDING'
    stable = True
    name = "argsort_matrix"

    input_dict = {
        "values": values,
        "axis": axis,
        "direction": direction,
        "stable": stable,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    values = tf.constant([1, 2, 3, 4, 5], dtype=tf.int64).numpy()
    axis = -1
    direction = 'ASCENDING'
    stable = False
    name = None

    input_dict = {
        "values": values,
        "axis": axis,
        "direction": direction,
        "stable": stable,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    values = tf.constant([[1, 2], [3, 4]], dtype=tf.float64).numpy()
    axis = 1
    direction = 'DESCENDING'
    stable = True
    name = 'argsort_2d'
    input_dict = {
        "values": values,
        "axis": axis,
        "direction": direction,
        "stable": stable,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    values = tf.constant([[-1, -2], [-3, -4]], dtype=tf.int32).numpy()
    axis = 0
    direction = 'ASCENDING'
    stable = False
    name = None
    input_dict = {
        "values": values,
        "axis": axis,
        "direction": direction,
        "stable": stable,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6
    values = tf.constant([5, 4, 3, 2, 1], dtype=tf.int32).numpy()
    axis = -1
    direction = 'DESCENDING'
    stable = True
    name = None
    input_dict = {
        "values": values,
        "axis": axis,
        "direction": direction,
        "stable": stable,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    values = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=tf.float32).numpy()
    axis = 2
    direction = 'ASCENDING'
    stable = False
    name = "3d_tensor"
    input_dict = {
        "values": values,
        "axis": axis,
        "direction": direction,
        "stable": stable,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    values = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=tf.int64).numpy()
    axis = 0
    direction = 'DESCENDING'
    stable = True
    name = "3d_tensor2"
    input_dict = {
        "values": values,
        "axis": axis,
        "direction": direction,
        "stable": stable,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    values = tf.constant([[1,1,1],[2,2,2],[3,3,3]], dtype=tf.int32).numpy()
    axis = 1
    direction = 'ASCENDING'
    stable = True
    name = "same_values"
    input_dict = {
        "values": values,
        "axis": axis,
        "direction": direction,
        "stable": stable,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 10
    values = tf.constant([1.0, 1.0, 1.0], dtype=tf.float32).numpy()
    axis = -1
    direction = 'ASCENDING'
    stable = False
    name = "same_values_1d"
    input_dict = {
        "values": values,
        "axis": axis,
        "direction": direction,
        "stable": stable,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.argsort"] = tf_argsort_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.argsort' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.argsort'.")

check_valid('tf.argsort', generated_inputs['tf.argsort'], lib="tf", suffix=0)
