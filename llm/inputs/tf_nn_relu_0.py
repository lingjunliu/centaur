
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_relu_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D float32 tensor
    features = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=tf.float32)
    name = "relu_1"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float64 tensor with negative and positive values
    features = tf.constant([[-1.5, 2.5], [0.0, -3.0]], dtype=tf.float64)
    name = "relu_2"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D int32 tensor with only positive values
    features = tf.constant([1, 2, 3, 4, 5], dtype=tf.int32)
    name = "relu_3"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D int64 tensor with mixed positive and negative values
    features = tf.constant([[-1, 2], [-3, 4]], dtype=tf.int64)
    name = "relu_4"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float32 tensor
    features = tf.constant([[[1.0, -2.0], [3.0, -4.0]], [[-5.0, 6.0], [-7.0, 8.0]]], dtype=tf.float32)
    name = "relu_5"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 1D uint8 tensor
    features = tf.constant([1, 2, 0, 255], dtype=tf.uint8)
    name = "relu_6"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty float32 tensor
    features = tf.constant([], dtype=tf.float32)
    name = "relu_7"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: scalar float32 tensor
    features = tf.constant(-5.0, dtype=tf.float32)
    name = "relu_8"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_nn_relu_inputs()
for input_dict in inputs:
    input_dict['features'] = input_dict['features'].numpy()

generated_inputs["tf.nn.relu"] = inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.relu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.relu'.")

check_valid('tf.nn.relu', generated_inputs['tf.nn.relu'], lib="tf", suffix=0)
