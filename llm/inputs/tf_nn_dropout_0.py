
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_dropout_inputs():
    list_of_inputs = []

    # Input 1
    x = np.ones((3, 5), dtype=np.float32)
    rate = 0.5
    noise_shape = None
    seed = 1
    name = "dropout_1"
    input_dict = {"x": x, "rate": rate, "noise_shape": noise_shape, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.random.rand(2, 2, 2).astype(np.float32)
    rate = 0.2
    noise_shape = None
    seed = 42
    name = "dropout_2"
    input_dict = {"x": x, "rate": rate, "noise_shape": noise_shape, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.random.rand(10).astype(np.float32)
    rate = 0.8
    noise_shape = None
    seed = 123
    name = "dropout_3"
    input_dict = {"x": x, "rate": rate, "noise_shape": noise_shape, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.random.rand(5, 5).astype(np.float32)
    rate = 0.0
    noise_shape = None
    seed = 99
    name = "dropout_4"
    input_dict = {"x": x, "rate": rate, "noise_shape": noise_shape, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.random.rand(3, 10).astype(np.float32)
    rate = 2/3
    noise_shape = np.array([1, 10], dtype=np.int32)
    seed = 1
    name = "dropout_5"
    input_dict = {"x": x, "rate": rate, "noise_shape": noise_shape, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = np.random.rand(2, 3, 4).astype(np.float32)
    rate = 0.3
    noise_shape = np.array([2, 1, 4], dtype=np.int32)
    seed = 7
    name = "dropout_6"
    input_dict = {"x": x, "rate": rate, "noise_shape": noise_shape, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = np.random.rand(4, 4).astype(np.float32)
    rate = 0.75
    noise_shape = np.array([4, 1], dtype=np.int32)
    seed = 15
    name = "dropout_7"
    input_dict = {"x": x, "rate": rate, "noise_shape": noise_shape, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = np.random.rand(1, 5, 1).astype(np.float32)
    rate = 0.6
    noise_shape = np.array([1, 5, 1], dtype=np.int32)
    seed = 23
    name = "dropout_8"
    input_dict = {"x": x, "rate": rate, "noise_shape": noise_shape, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = np.random.rand(8).astype(np.float32)
    rate = 0.9
    noise_shape = np.array([8], dtype=np.int32)
    seed = 31
    name = "dropout_9"
    input_dict = {"x": x, "rate": rate, "noise_shape": noise_shape, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = np.random.rand(1, 1, 1, 1).astype(np.float32)
    rate = 0.4
    noise_shape = None
    seed = 39
    name = "dropout_10"
    input_dict = {"x": x, "rate": rate, "noise_shape": noise_shape, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.dropout"] = tf_nn_dropout_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.dropout' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.dropout'.")

check_valid('tf.nn.dropout', generated_inputs['tf.nn.dropout'], lib="tf", suffix=0)
