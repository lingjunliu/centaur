
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_broadcast_static_shape_inputs():
    list_of_inputs = []

    # Input 1
    shape_x = np.array([1, 2, 3], dtype=np.int32)
    shape_y = np.array([5, 1, 3], dtype=np.int32)
    input_dict = {"shape_x": shape_x, "shape_y": shape_y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    shape_x = np.array([1, 2, 3], dtype=np.int32)
    shape_y = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {"shape_x": shape_x, "shape_y": shape_y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    shape_x = np.array([5, 2, 3], dtype=np.int32)
    shape_y = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {"shape_x": shape_x, "shape_y": shape_y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    shape_x = np.array([5, 2, 1], dtype=np.int32)
    shape_y = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {"shape_x": shape_x, "shape_y": shape_y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    shape_x = np.array([2, 1], dtype=np.int32)
    shape_y = np.array([1, 3], dtype=np.int32)
    input_dict = {"shape_x": shape_x, "shape_y": shape_y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    shape_x = np.array([1], dtype=np.int32)
    shape_y = np.array([5], dtype=np.int32)
    input_dict = {"shape_x": shape_x, "shape_y": shape_y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    shape_x = np.array([], dtype=np.int32)
    shape_y = np.array([], dtype=np.int32)
    input_dict = {"shape_x": shape_x, "shape_y": shape_y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    shape_x = np.array([2, 3, 4, 5], dtype=np.int32)
    shape_y = np.array([1, 3, 1, 5], dtype=np.int32)
    input_dict = {"shape_x": shape_x, "shape_y": shape_y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    shape_x = np.array([2, 3, 4, 5], dtype=np.int32)
    shape_y = np.array([2, 3, 4, 5], dtype=np.int32)
    input_dict = {"shape_x": shape_x, "shape_y": shape_y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    shape_x = np.array([1, 5, 4], dtype=np.int32)
    shape_y = np.array([5, 1, 4], dtype=np.int32)
    input_dict = {"shape_x": shape_x, "shape_y": shape_y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_broadcast_static_shape_inputs()
generated_inputs["tf.broadcast_static_shape"] = inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.broadcast_static_shape' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.broadcast_static_shape'.")

check_valid('tf.broadcast_static_shape', generated_inputs['tf.broadcast_static_shape'], lib="tf", suffix=0)
