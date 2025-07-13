
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_truncatemod_inputs():
    list_of_inputs = []

    # Input 1: Basic integers
    x = np.array([10, 15, 20], dtype=np.int32)
    y = np.array([3, 4, 6], dtype=np.int32)
    name = "basic_integers"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative integers
    x = np.array([-10, -15, -20], dtype=np.int64)
    y = np.array([3, -4, 6], dtype=np.int64)
    name = "negative_integers"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Floats
    x = np.array([10.5, 15.2, 20.7], dtype=np.float32)
    y = np.array([3.0, 4.0, 6.0], dtype=np.float32)
    name = "floats"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative floats
    x = np.array([-10.5, -15.2, -20.7], dtype=np.float64)
    y = np.array([3.0, -4.0, 6.0], dtype=np.float64)
    name = "negative_floats"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broadcasting with integers
    x = np.array([10, 15, 20], dtype=np.int32)
    y = np.array([5], dtype=np.int32)
    name = "broadcasting_int"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Broadcasting with floats
    x = np.array([10.5, 15.2, 20.7], dtype=np.float32)
    y = np.array([2.0], dtype=np.float32)
    name = "broadcasting_float"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Multi-dimensional integers
    x = np.array([[10, 15], [20, 25]], dtype=np.int64)
    y = np.array([[3, 4], [6, 7]], dtype=np.int64)
    name = "multi_dim_int"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Multi-dimensional floats
    x = np.array([[10.5, 15.2], [20.7, 25.4]], dtype=np.float64)
    y = np.array([[3.0, 4.0], [6.0, 7.0]], dtype=np.float64)
    name = "multi_dim_float"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9: Remove bfloat16 to avoid errors.

    # Input 10: Remove half to avoid errors.

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.truncatemod"] = tf_truncatemod_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.truncatemod' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.truncatemod'.")

check_valid('tf.truncatemod', generated_inputs['tf.truncatemod'], lib="tf", suffix=0)
