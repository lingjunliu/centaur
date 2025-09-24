
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_floor_inputs():
    list_of_inputs = []

    # Input 1: float32, 1D array
    x = np.array([1.5, 2.7, -3.2, 0.0], dtype=np.float32)
    input_dict = {"x": x, "name": "floor_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, 2D array
    x = np.array([[1.1, 2.2], [3.3, 4.4], [-5.5, -6.6]], dtype=np.float64)
    input_dict = {"x": x, "name": "floor_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32, scalar
    x = np.array(7.8, dtype=np.float32)
    input_dict = {"x": x, "name": "floor_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float32, 3D array
    x = np.array([[[1.6, 2.8], [3.9, 4.1]], [[5.2, 6.3], [7.4, 8.5]]], dtype=np.float32)
    input_dict = {"x": x, "name": "floor_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32, 0D array
    x = np.array(-4.9, dtype=np.float32)
    input_dict = {"x": x, "name": "floor_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float64, 1D array with negative and positive values, zeros
    x = np.array([-1.5, 0.0, 2.7, -0.0, 3.2], dtype=np.float64)
    input_dict = {"x": x, "name": "floor_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32, 2D array with large values
    x = np.array([[1000.5, 2000.7], [-3000.2, 4000.0]], dtype=np.float32)
    input_dict = {"x": x, "name": "floor_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float32, 1D array with small values
    x = np.array([0.1, 0.2, -0.3, -0.4], dtype=np.float32)
    input_dict = {"x": x, "name": "floor_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32, 2D with -1.0 and 1.0 values
    x = np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=np.float32)
    input_dict = {"x": x, "name": "floor_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float64, 3D array with mixed values
    x = np.array([[[1.5, -2.7], [3.2, 0.0]], [[-1.1, 2.2], [-3.3, 4.4]]], dtype=np.float64)
    input_dict = {"x": x, "name": "floor_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Floor"] = tf_raw_ops_floor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Floor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Floor'.")

check_valid('tf.raw_ops.Floor', generated_inputs['tf.raw_ops.Floor'], lib="tf", suffix=0)
