
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_atan2_inputs():
    list_of_inputs = []

    # Input 1
    y = np.array([1.0, 1.0]).astype(np.float32)
    x = np.array([1.0, -1.0]).astype(np.float32)
    name = "atan2_1"
    input_dict = {"y": y, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    y = np.array([-1.0, -1.0]).astype(np.float32)
    x = np.array([1.0, -1.0]).astype(np.float32)
    name = "atan2_2"
    input_dict = {"y": y, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    y = np.array([0.0, 0.0]).astype(np.float32)
    x = np.array([1.0, -1.0]).astype(np.float32)
    name = "atan2_3"
    input_dict = {"y": y, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    y = np.array([1.0, -1.0]).astype(np.float32)
    x = np.array([0.0, 0.0]).astype(np.float32)
    name = "atan2_4"
    input_dict = {"y": y, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    y = np.array([[1.0, 2.0], [3.0, 4.0]]).astype(np.float32)
    x = np.array([[5.0, 6.0], [7.0, 8.0]]).astype(np.float32)
    name = "atan2_5"
    input_dict = {"y": y, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    y = np.array([[-1.0, -2.0], [-3.0, -4.0]]).astype(np.float32)
    x = np.array([[5.0, 6.0], [7.0, 8.0]]).astype(np.float32)
    name = "atan2_6"
    input_dict = {"y": y, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    y = np.array([1.0, 2.0, 3.0]).astype(np.float32)
    x = np.array([-1.0, -2.0, -3.0]).astype(np.float32)
    name = "atan2_7"
    input_dict = {"y": y, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    y = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).astype(np.float32)
    x = np.array([[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]]).astype(np.float32)
    name = "atan2_8"
    input_dict = {"y": y, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    y = np.array([1.0]).astype(np.float32)
    x = np.array([1.0]).astype(np.float32)
    name = "atan2_9"
    input_dict = {"y": y, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    y = np.array([[-1.0]]).astype(np.float32)
    x = np.array([[1.0]]).astype(np.float32)
    name = "atan2_10"
    input_dict = {"y": y, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.atan2"] = tf_math_atan2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.atan2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.atan2'.")

check_valid('tf.math.atan2', generated_inputs['tf.math.atan2'], lib="tf", suffix=0)
