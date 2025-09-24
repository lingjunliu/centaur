
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_l2_normalize_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array([3.0, 4.0], dtype=np.float32)
    axis = 0
    epsilon = 1e-12
    name = "l2_norm_1"
    input_dict = {"x": x, "axis": axis, "epsilon": epsilon, "name": name, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array([[3.0], [4.0]], dtype=np.float32)
    axis = 0
    epsilon = 1e-12
    name = "l2_norm_2"
    input_dict = {"x": x, "axis": axis, "epsilon": epsilon, "name": name, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.array([[3.0, 1.0], [4.0, 2.0]], dtype=np.float32)
    axis = 1
    epsilon = 1e-12
    name = "l2_norm_3"
    input_dict = {"x": x, "axis": axis, "epsilon": epsilon, "name": name, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.array([[-3.0, 1.0], [-4.0, 2.0]], dtype=np.float32)
    axis = 0
    epsilon = 1e-12
    name = "l2_norm_4"
    input_dict = {"x": x, "axis": axis, "epsilon": epsilon, "name": name, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    axis = 2
    epsilon = 1e-12
    name = "l2_norm_5"
    input_dict = {"x": x, "axis": axis, "epsilon": epsilon, "name": name, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    axis = 0
    epsilon = 1e-6
    name = "l2_norm_6"
    input_dict = {"x": x, "axis": axis, "epsilon": epsilon, "name": name, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    axis = 0
    epsilon = 1e-9
    name = "l2_norm_7"
    input_dict = {"x": x, "axis": axis, "epsilon": epsilon, "name": name, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    axis = 0
    epsilon = 0.0
    name = "l2_norm_8"
    input_dict = {"x": x, "axis": axis, "epsilon": epsilon, "name": name, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    axis = 0
    epsilon = 0.0
    name = "l2_norm_9"
    input_dict = {"x": x, "axis": axis, "epsilon": epsilon, "name": name, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    axis = 0
    epsilon = 1e-12
    name = "l2_norm_10"
    input_dict = {"x": x, "axis": axis, "epsilon": epsilon, "name": name, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.l2_normalize"] = tf_math_l2_normalize_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.l2_normalize' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.l2_normalize'.")

check_valid('tf.math.l2_normalize', generated_inputs['tf.math.l2_normalize'], lib="tf", suffix=0)
