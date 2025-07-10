
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_sufficient_statistics_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    axes = [0]
    shift = np.array(0.0, dtype=np.float32)
    keepdims = False
    name = "sufficient_stats_1"
    input_dict = {"x": tf.constant(x), "axes": axes, "shift": tf.constant(shift), "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    axes = [0, 1]
    shift = np.array(2.0, dtype=np.float32)
    keepdims = True
    name = "sufficient_stats_2"
    input_dict = {"x": tf.constant(x), "axes": axes, "shift": tf.constant(shift), "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    axes = [0]
    shift = np.array(1.5, dtype=np.float32)
    keepdims = False
    name = "sufficient_stats_3"
    input_dict = {"x": tf.constant(x), "axes": axes, "shift": tf.constant(shift), "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    axes = [0, 1]
    shift = np.array(4.0, dtype=np.float32)
    keepdims = False
    name = "sufficient_stats_4"
    input_dict = {"x": tf.constant(x), "axes": axes, "shift": tf.constant(shift), "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.array([1, 2, 3, 4], dtype=np.int32).astype(np.float32)
    axes = [0]
    shift = np.array(2.0, dtype=np.float32)
    keepdims = False
    name = "sufficient_stats_5"
    input_dict = {"x": tf.constant(x), "axes": axes, "shift": tf.constant(shift), "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = np.array([-1.0, -2.0, -3.0, -4.0], dtype=np.float32)
    axes = [0]
    shift = np.array(-2.5, dtype=np.float32)
    keepdims = False
    name = "sufficient_stats_6"
    input_dict = {"x": tf.constant(x), "axes": axes, "shift": tf.constant(shift), "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    axes = [1]
    shift = np.array(0.0, dtype=np.float32)
    keepdims = False
    name = "sufficient_stats_7"
    input_dict = {"x": tf.constant(x), "axes": axes, "shift": tf.constant(shift), "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    axes = [0]
    shift = np.array(0.0, dtype=np.float32)
    keepdims = False
    name = "sufficient_stats_8"
    input_dict = {"x": tf.constant(x), "axes": axes, "shift": tf.constant(shift), "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    axes = [1]
    shift = np.array(0.0, dtype=np.float32)
    keepdims = False
    name = "sufficient_stats_9"
    input_dict = {"x": tf.constant(x), "axes": axes, "shift": tf.constant(shift), "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    axes = [2]
    shift = np.array(0.0, dtype=np.float32)
    keepdims = False
    name = "sufficient_stats_10"
    input_dict = {"x": tf.constant(x), "axes": axes, "shift": tf.constant(shift), "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.sufficient_statistics"] = tf_nn_sufficient_statistics_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.sufficient_statistics' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.sufficient_statistics'.")

check_valid('tf.nn.sufficient_statistics', generated_inputs['tf.nn.sufficient_statistics'], lib="tf", suffix=0)
