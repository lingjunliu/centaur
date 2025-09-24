
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_local_response_normalization_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.random.rand(1, 5, 5, 3).astype(np.float32)
    depth_radius = 2
    bias = 1.0
    alpha = 1.0
    beta = 0.5
    name = "lrn1"
    input_dict = {"input": input_tensor, "depth_radius": depth_radius, "bias": bias, "alpha": alpha, "beta": beta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.random.rand(2, 10, 10, 5).astype(np.float32)
    depth_radius = 3
    bias = 2.0
    alpha = 0.5
    beta = 0.2
    name = "lrn2"
    input_dict = {"input": input_tensor, "depth_radius": depth_radius, "bias": bias, "alpha": alpha, "beta": beta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.random.rand(1, 8, 8, 1).astype(np.float32)
    depth_radius = 1
    bias = 0.5
    alpha = 2.0
    beta = 1.0
    name = "lrn3"
    input_dict = {"input": input_tensor, "depth_radius": depth_radius, "bias": bias, "alpha": alpha, "beta": beta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.random.rand(4, 12, 12, 7).astype(np.float32)
    depth_radius = 4
    bias = 1.5
    alpha = 0.8
    beta = 0.7
    name = "lrn4"
    input_dict = {"input": input_tensor, "depth_radius": depth_radius, "bias": bias, "alpha": alpha, "beta": beta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    input_tensor = np.random.rand(1, 3, 3, 2).astype(np.float32)
    depth_radius = 5
    bias = 1.0
    alpha = 1.0
    beta = 0.5
    name = "lrn5"
    input_dict = {"input": input_tensor, "depth_radius": depth_radius, "bias": bias, "alpha": alpha, "beta": beta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.random.rand(1, 5, 5, 3).astype(np.float32)
    depth_radius = 0
    bias = 1.0
    alpha = 1.0
    beta = 0.5
    name = "lrn6"
    input_dict = {"input": input_tensor, "depth_radius": depth_radius, "bias": bias, "alpha": alpha, "beta": beta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.random.rand(1, 5, 5, 3).astype(np.float32)
    depth_radius = 2
    bias = 0.0
    alpha = 1.0
    beta = 0.5
    name = "lrn7"
    input_dict = {"input": input_tensor, "depth_radius": depth_radius, "bias": bias, "alpha": alpha, "beta": beta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8: bfloat16
    input_tensor = np.random.rand(1, 5, 5, 3).astype(np.float32)
    depth_radius = 2
    bias = 1.0
    alpha = 1.0
    beta = 0.5
    name = "lrn8"
    input_dict = {"input": input_tensor, "depth_radius": depth_radius, "bias": bias, "alpha": alpha, "beta": beta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Smaller values
    input_tensor = (np.random.rand(1, 5, 5, 3) * 0.1).astype(np.float32)
    depth_radius = 2
    bias = 1.0
    alpha = 1.0
    beta = 0.5
    name = "lrn9"
    input_dict = {"input": input_tensor, "depth_radius": depth_radius, "bias": bias, "alpha": alpha, "beta": beta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Different dimensions
    input_tensor = np.random.rand(3, 7, 7, 4).astype(np.float32)
    depth_radius = 3
    bias = 1.0
    alpha = 1.0
    beta = 0.5
    name = "lrn10"
    input_dict = {"input": input_tensor, "depth_radius": depth_radius, "bias": bias, "alpha": alpha, "beta": beta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.local_response_normalization"] = tf_nn_local_response_normalization_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.local_response_normalization' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.local_response_normalization'.")

check_valid('tf.nn.local_response_normalization', generated_inputs['tf.nn.local_response_normalization'], lib="tf", suffix=0)
