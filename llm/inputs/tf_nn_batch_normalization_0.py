
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_batch_normalization_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D input
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    mean = np.array([2.0], dtype=np.float32)
    variance = np.array([1.0], dtype=np.float32)
    offset = np.array([0.0], dtype=np.float32)
    scale = np.array([1.0], dtype=np.float32)
    variance_epsilon = 0.001
    name = "batch_norm_1"

    input_dict = {
        "x": x,
        "mean": mean,
        "variance": variance,
        "offset": offset,
        "scale": scale,
        "variance_epsilon": variance_epsilon,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic 2D input
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mean = np.array([2.0, 3.0], dtype=np.float32)
    variance = np.array([1.0, 1.0], dtype=np.float32)
    offset = np.array([0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0], dtype=np.float32)
    variance_epsilon = 0.001
    name = "batch_norm_2"

    input_dict = {
        "x": x,
        "mean": mean,
        "variance": variance,
        "offset": offset,
        "scale": scale,
        "variance_epsilon": variance_epsilon,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D input with different shapes
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    mean = np.array([4.0, 5.0], dtype=np.float32)
    variance = np.array([2.0, 2.0], dtype=np.float32)
    offset = np.array([0.5, 0.5], dtype=np.float32)
    scale = np.array([0.8, 0.8], dtype=np.float32)
    variance_epsilon = 0.001
    name = "batch_norm_3"

    input_dict = {
        "x": x,
        "mean": mean,
        "variance": variance,
        "offset": offset,
        "scale": scale,
        "variance_epsilon": variance_epsilon,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative values
    x = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    mean = np.array([-2.0], dtype=np.float32)
    variance = np.array([1.0], dtype=np.float32)
    offset = np.array([0.0], dtype=np.float32)
    scale = np.array([1.0], dtype=np.float32)
    variance_epsilon = 0.001
    name = "batch_norm_4"

    input_dict = {
        "x": x,
        "mean": mean,
        "variance": variance,
        "offset": offset,
        "scale": scale,
        "variance_epsilon": variance_epsilon,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Zero variance
    x = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    mean = np.array([1.0], dtype=np.float32)
    variance = np.array([0.0], dtype=np.float32)
    offset = np.array([0.0], dtype=np.float32)
    scale = np.array([1.0], dtype=np.float32)
    variance_epsilon = 0.001
    name = "batch_norm_5"

    input_dict = {
        "x": x,
        "mean": mean,
        "variance": variance,
        "offset": offset,
        "scale": scale,
        "variance_epsilon": variance_epsilon,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6: Different scale and offset
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    mean = np.array([2.0], dtype=np.float32)
    variance = np.array([1.0], dtype=np.float32)
    offset = np.array([0.5], dtype=np.float32)
    scale = np.array([2.0], dtype=np.float32)
    variance_epsilon = 0.001
    name = "batch_norm_6"

    input_dict = {
        "x": x,
        "mean": mean,
        "variance": variance,
        "offset": offset,
        "scale": scale,
        "variance_epsilon": variance_epsilon,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger variance epsilon
    x = np.array([1.0, 2.0], dtype=np.float32)
    mean = np.array([1.5], dtype=np.float32)
    variance = np.array([0.25], dtype=np.float32)
    offset = np.array([0.0], dtype=np.float32)
    scale = np.array([1.0], dtype=np.float32)
    variance_epsilon = 0.1
    name = "batch_norm_7"

    input_dict = {
        "x": x,
        "mean": mean,
        "variance": variance,
        "offset": offset,
        "scale": scale,
        "variance_epsilon": variance_epsilon,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 4D input
    x = np.random.rand(2, 3, 4, 5).astype(np.float32)
    mean = np.random.rand(5).astype(np.float32)
    variance = np.random.rand(5).astype(np.float32)
    offset = np.random.rand(5).astype(np.float32)
    scale = np.random.rand(5).astype(np.float32)
    variance_epsilon = 0.001
    name = "batch_norm_8"

    input_dict = {
        "x": x,
        "mean": mean,
        "variance": variance,
        "offset": offset,
        "scale": scale,
        "variance_epsilon": variance_epsilon,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D input with different mean, variance, offset, scale
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    mean = np.array([3.0], dtype=np.float32)
    variance = np.array([2.0], dtype=np.float32)
    offset = np.array([0.5], dtype=np.float32)
    scale = np.array([1.5], dtype=np.float32)
    variance_epsilon = 0.01
    name = "batch_norm_9"

    input_dict = {
        "x": x,
        "mean": mean,
        "variance": variance,
        "offset": offset,
        "scale": scale,
        "variance_epsilon": variance_epsilon,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Mean and variance 2D with keepdims=True equivalent shape
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mean = np.array([[2.0], [3.0]], dtype=np.float32)
    variance = np.array([[1.0], [1.0]], dtype=np.float32)
    offset = np.array([[0.0], [0.0]], dtype=np.float32)
    scale = np.array([[1.0], [1.0]], dtype=np.float32)
    variance_epsilon = 0.001
    name = "batch_norm_10"

    input_dict = {
        "x": x,
        "mean": mean,
        "variance": variance,
        "offset": offset,
        "scale": scale,
        "variance_epsilon": variance_epsilon,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.batch_normalization"] = tf_nn_batch_normalization_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.batch_normalization' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.batch_normalization'.")

check_valid('tf.nn.batch_normalization', generated_inputs['tf.nn.batch_normalization'], lib="tf", suffix=0)
