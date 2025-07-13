
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_atan2_inputs():
    list_of_inputs = []

    # Input 1: Basic positive values
    y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    name = "atan2_basic_pos"
    input_dict = {"y": y, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative values
    y = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    x = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    name = "atan2_negative_y"
    input_dict = {"y": y, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Mixed positive and negative
    y = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    x = np.array([-1.0, 1.0, -3.0], dtype=np.float32)
    name = "atan2_mixed"
    input_dict = {"y": y, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Zero values
    y = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    x = np.array([1.0, -1.0, 0.0], dtype=np.float32)
    name = "atan2_zero_y"
    input_dict = {"y": y, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D arrays
    y = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    x = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    name = "atan2_2d"
    input_dict = {"y": y, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different float types
    y = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    x = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    name = "atan2_float64"
    input_dict = {"y": y, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: x values of zero, y positive
    y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    name = "atan2_x_zero_y_pos"
    input_dict = {"y": y, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: x values of zero, y negative
    y = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    x = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    name = "atan2_x_zero_y_neg"
    input_dict = {"y": y, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float16
    y = np.array([1.0, 2.0], dtype=np.float16)
    x = np.array([1.0, 1.0], dtype=np.float16)
    name = "atan2_half"
    input_dict = {"y": y, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different shapes
    y = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    x = np.array([[[0.5, 1.0], [1.5, 2.0]], [[2.5, 3.0], [3.5, 4.0]]], dtype=np.float32)
    name = "atan2_3d"
    input_dict = {"y": y, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: All zeros
    y = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    x = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    name = "atan2_all_zeros"
    input_dict = {"y": y, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Atan2"] = tf_raw_ops_atan2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Atan2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Atan2'.")

check_valid('tf.raw_ops.Atan2', generated_inputs['tf.raw_ops.Atan2'], lib="tf", suffix=0)
