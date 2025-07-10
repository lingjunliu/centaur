
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_relu6_inputs():
    list_of_inputs = []

    # Input 1: Basic float32
    features = tf.constant([-3.0, -1.0, 0.0, 6.0, 10.0], dtype=tf.float32).numpy()
    name = None
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: int32 with name
    features = tf.constant([-3, -1, 0, 6, 10], dtype=tf.int32).numpy()
    name = "relu6_example_int32"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64
    features = tf.constant([-3.0, -1.0, 0.0, 6.0, 10.0], dtype=tf.float64).numpy()
    name = None
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: uint8
    features = tf.constant([252, 254, 0, 6, 10], dtype=tf.uint8).numpy()
    name = None
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: int16
    features = tf.constant([-3, -1, 0, 6, 10], dtype=tf.int16).numpy()
    name = None
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: int8
    features = tf.constant([-3, -1, 0, 6, 10], dtype=tf.int8).numpy()
    name = None
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 2D int64
    features = tf.constant([[-3, -1], [0, 6], [10, -2]], dtype=tf.int64).numpy()
    name = None
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D float32 with larger values
    features = tf.constant([[[ -7.0, 1.0], [5.0, 100.0]], [[-1.0, -2.0], [6.0, 7.0]]], dtype=tf.float32).numpy()
    name = None
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: All negative int32
    features = tf.constant([-1, -2, -3, -4, -5], dtype=tf.int32).numpy()
    name = "all_negative"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: All positive int32
    features = tf.constant([1, 2, 3, 4, 5], dtype=tf.int32).numpy()
    name = "all_positive"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.relu6"] = tf_nn_relu6_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.relu6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.relu6'.")

check_valid('tf.nn.relu6', generated_inputs['tf.nn.relu6'], lib="tf", suffix=0)
