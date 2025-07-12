
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_weighted_moments_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    axes = np.array([0], dtype=np.int32)
    frequency_weights = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    keepdims = False
    name = "moments_1"
    input_dict = {"x": x, "axes": axes.tolist(), "frequency_weights": frequency_weights, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    axes = np.array([0, 1], dtype=np.int32)
    frequency_weights = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    keepdims = True
    name = "moments_2"
    input_dict = {"x": x, "axes": axes.tolist(), "frequency_weights": frequency_weights, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    axes = np.array([0], dtype=np.int32)
    frequency_weights = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    keepdims = False
    name = "moments_3"
    input_dict = {"x": x, "axes": axes.tolist(), "frequency_weights": frequency_weights, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    axes = np.array([0], dtype=np.int32)
    frequency_weights = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    keepdims = False
    name = "moments_4"
    input_dict = {"x": x, "axes": axes.tolist(), "frequency_weights": frequency_weights, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    axes = np.array([0, 1, 2], dtype=np.int32)
    frequency_weights = np.array([[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]], dtype=np.float32)
    keepdims = False
    name = "moments_5"
    input_dict = {"x": x, "axes": axes.tolist(), "frequency_weights": frequency_weights, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    axes = np.array([0], dtype=np.int32)
    frequency_weights = np.array([2.0, 1.0, 0.5], dtype=np.float32)
    keepdims = True
    name = "moments_6"
    input_dict = {"x": x, "axes": axes.tolist(), "frequency_weights": frequency_weights, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    axes = np.array([1], dtype=np.int32)
    frequency_weights = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float32)
    keepdims = False
    name = "moments_7"
    input_dict = {"x": x, "axes": axes.tolist(), "frequency_weights": frequency_weights, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    axes = np.array([0], dtype=np.int32)
    frequency_weights = np.array([[[1.0, 1.0], [1.0, 1.0]], [[0.5, 0.5], [0.5, 0.5]]], dtype=np.float32)
    keepdims = True
    name = "moments_8"
    input_dict = {"x": x, "axes": axes.tolist(), "frequency_weights": frequency_weights, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    axes = np.array([0], dtype=np.int32)
    frequency_weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    keepdims = False
    name = "moments_9"
    input_dict = {"x": x, "axes": axes.tolist(), "frequency_weights": frequency_weights, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    axes = np.array([0, 1], dtype=np.int32)
    frequency_weights = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    keepdims = False
    name = "moments_10"
    input_dict = {"x": x, "axes": axes.tolist(), "frequency_weights": frequency_weights, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.weighted_moments"] = tf_nn_weighted_moments_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.weighted_moments' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.weighted_moments'.")

check_valid('tf.nn.weighted_moments', generated_inputs['tf.nn.weighted_moments'], lib="tf", suffix=0)
