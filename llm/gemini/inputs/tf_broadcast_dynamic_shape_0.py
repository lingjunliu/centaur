
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_broadcast_dynamic_shape_inputs():
    list_of_inputs = []

    # Input 1: Simple case
    shape_x = np.array([1, 2, 3], dtype=np.int32)
    shape_y = np.array([5, 1, 3], dtype=np.int32)
    input_dict = {"shape_x": shape_x, "shape_y": shape_y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Broadcasting a scalar
    shape_x = np.array([1], dtype=np.int32)
    shape_y = np.array([5, 4, 3], dtype=np.int32)
    input_dict = {"shape_x": shape_x, "shape_y": shape_y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Broadcasting with same shapes
    shape_x = np.array([2, 3, 4], dtype=np.int32)
    shape_y = np.array([2, 3, 4], dtype=np.int32)
    input_dict = {"shape_x": shape_x, "shape_y": shape_y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcasting with leading 1s
    shape_x = np.array([1, 5, 3], dtype=np.int32)
    shape_y = np.array([5, 1, 3], dtype=np.int32)
    input_dict = {"shape_x": shape_x, "shape_y": shape_y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different lengths but compatible
    shape_x = np.array([3], dtype=np.int32)
    shape_y = np.array([5, 1, 3], dtype=np.int32)
    input_dict = {"shape_x": shape_x, "shape_y": shape_y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: More dimensions
    shape_x = np.array([1, 4, 1, 2], dtype=np.int32)
    shape_y = np.array([3, 1, 5, 2], dtype=np.int32)
    input_dict = {"shape_x": shape_x, "shape_y": shape_y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different dtypes, explicitly specified
    shape_x = np.array([1, 2, 3], dtype=np.int64).astype(np.int32)
    shape_y = np.array([5, 1, 3], dtype=np.int64).astype(np.int32)
    input_dict = {"shape_x": shape_x, "shape_y": shape_y}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.broadcast_dynamic_shape"] = tf_broadcast_dynamic_shape_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.broadcast_dynamic_shape' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.broadcast_dynamic_shape'.")

check_valid('tf.broadcast_dynamic_shape', generated_inputs['tf.broadcast_dynamic_shape'], lib="tf", suffix=0)
