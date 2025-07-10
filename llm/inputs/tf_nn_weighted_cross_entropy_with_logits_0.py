
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_weighted_cross_entropy_with_logits_inputs():
    list_of_inputs = []

    # Input 1
    labels = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    logits = np.array([2.0, -1.0, 0.5], dtype=np.float32)
    pos_weight = np.array(2.0, dtype=np.float32)
    name = "example1"
    input_dict = {"labels": labels, "logits": logits, "pos_weight": pos_weight, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    labels = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    logits = np.array([[-1.0, 2.0], [0.5, -0.5]], dtype=np.float32)
    pos_weight = np.array(0.5, dtype=np.float32)
    name = "example2"
    input_dict = {"labels": labels, "logits": logits, "pos_weight": pos_weight, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    labels = np.array([0.2, 0.8, 0.5], dtype=np.float32)
    logits = np.array([-2.0, 1.5, -0.1], dtype=np.float32)
    pos_weight = np.array(1.0, dtype=np.float32)
    name = "example3"
    input_dict = {"labels": labels, "logits": logits, "pos_weight": pos_weight, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    labels = np.array([[[1.0, 0.0], [0.5, 0.5]], [[0.0, 1.0], [0.2, 0.8]]], dtype=np.float32)
    logits = np.array([[[2.0, -1.0], [0.0, 1.0]], [[-1.5, 2.5], [-0.3, 0.7]]], dtype=np.float32)
    pos_weight = np.array(1.5, dtype=np.float32)
    name = "example4"
    input_dict = {"labels": labels, "logits": logits, "pos_weight": pos_weight, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    labels = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    logits = np.array([-5.0, -2.0, -1.0], dtype=np.float32)
    pos_weight = np.array(0.1, dtype=np.float32)
    name = "example5"
    input_dict = {"labels": labels, "logits": logits, "pos_weight": pos_weight, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    labels = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    logits = np.array([5.0, 2.0, 1.0], dtype=np.float32)
    pos_weight = np.array(5.0, dtype=np.float32)
    name = "example6"
    input_dict = {"labels": labels, "logits": logits, "pos_weight": pos_weight, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    labels = np.array([0.5], dtype=np.float64)
    logits = np.array([-0.5], dtype=np.float64)
    pos_weight = np.array(0.75, dtype=np.float64)
    name = "example7"
    input_dict = {"labels": labels, "logits": logits, "pos_weight": pos_weight, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    labels = np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float64)
    logits = np.array([[-2.1, 1.9], [1.2, -1.8]], dtype=np.float64)
    pos_weight = np.array(2.2, dtype=np.float64)
    name = "example8"
    input_dict = {"labels": labels, "logits": logits, "pos_weight": pos_weight, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    labels = np.array([0.3, 0.7], dtype=np.float32)
    logits = np.array([-1.3, 0.7], dtype=np.float32)
    pos_weight = np.array(3.0, dtype=np.float32)
    name = "example9"
    input_dict = {"labels": labels, "logits": logits, "pos_weight": pos_weight, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    labels = np.array([1.0, 0.0], dtype=np.float32)
    logits = np.array([0.0, 0.0], dtype=np.float32)
    pos_weight = np.array(1.0, dtype=np.float32)
    name = "example10"
    input_dict = {"labels": labels, "logits": logits, "pos_weight": pos_weight, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.weighted_cross_entropy_with_logits"] = tf_nn_weighted_cross_entropy_with_logits_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.weighted_cross_entropy_with_logits' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.weighted_cross_entropy_with_logits'.")

check_valid('tf.nn.weighted_cross_entropy_with_logits', generated_inputs['tf.nn.weighted_cross_entropy_with_logits'], lib="tf", suffix=0)
