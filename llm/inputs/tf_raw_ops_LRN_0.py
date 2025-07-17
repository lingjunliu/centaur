
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_lrn_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[[[1.0, 2.0, 3.0, 4.0]]]], dtype=np.float32)
    depth_radius = 2
    bias = 1.0
    alpha = 1.0
    beta = 0.5
    name = "lrn_1"
    input_dict = {"input": input_tensor, "depth_radius": depth_radius, "bias": bias, "alpha": alpha, "beta": beta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[[[1.0, 2.0, 3.0, 4.0]]]], dtype=np.float32)
    depth_radius = 1
    bias = 0.5
    alpha = 2.0
    beta = 0.2
    name = "lrn_2"
    input_dict = {"input": input_tensor, "depth_radius": depth_radius, "bias": bias, "alpha": alpha, "beta": beta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]], dtype=np.float32)
    depth_radius = 2
    bias = 1.0
    alpha = 1.0
    beta = 0.5
    name = "lrn_3"
    input_dict = {"input": input_tensor, "depth_radius": depth_radius, "bias": bias, "alpha": alpha, "beta": beta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]], dtype=np.float32)
    depth_radius = 3
    bias = 2.0
    alpha = 0.5
    beta = 1.0
    name = "lrn_4"
    input_dict = {"input": input_tensor, "depth_radius": depth_radius, "bias": bias, "alpha": alpha, "beta": beta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]], dtype=np.float32)
    depth_radius = 0
    bias = 1.0
    alpha = 1.0
    beta = 0.5
    name = "lrn_5"
    input_dict = {"input": input_tensor, "depth_radius": depth_radius, "bias": bias, "alpha": alpha, "beta": beta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]], dtype=np.float32)
    depth_radius = 5
    bias = 0.1
    alpha = 1.5
    beta = 0.75
    name = "lrn_6"
    input_dict = {"input": input_tensor, "depth_radius": depth_radius, "bias": bias, "alpha": alpha, "beta": beta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]], dtype=np.float32)
    depth_radius = 1
    bias = 1.0
    alpha = 1.0
    beta = 0.5
    name = "lrn_7"
    input_dict = {"input": input_tensor, "depth_radius": depth_radius, "bias": bias, "alpha": alpha, "beta": beta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]], dtype=np.float32)
    depth_radius = 2
    bias = 0.25
    alpha = 0.75
    beta = 0.25
    name = "lrn_8"
    input_dict = {"input": input_tensor, "depth_radius": depth_radius, "bias": bias, "alpha": alpha, "beta": beta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([[[[1.0], [2.0]], [[3.0], [4.0]]]], dtype=np.float32)
    depth_radius = 1
    bias = 1.0
    alpha = 1.0
    beta = 0.5
    name = "lrn_9"
    input_dict = {"input": input_tensor, "depth_radius": depth_radius, "bias": bias, "alpha": alpha, "beta": beta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 10
    input_tensor = np.array([[[[1.0], [2.0]], [[3.0], [4.0]]]], dtype=np.float32)
    depth_radius = 5
    bias = 1.5
    alpha = 0.8
    beta = 0.3
    name = "lrn_10"
    input_dict = {"input": input_tensor, "depth_radius": depth_radius, "bias": bias, "alpha": alpha, "beta": beta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.LRN"] = tf_raw_ops_lrn_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.LRN' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.LRN'.")

check_valid('tf.raw_ops.LRN', generated_inputs['tf.raw_ops.LRN'], lib="tf", suffix=0)
