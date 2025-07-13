
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_lrn_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.random.rand(1, 5, 5, 3).astype(np.float32)
    depth_radius = 2
    bias = 1.0
    alpha = 1.0
    beta = 0.5
    name = "lrn_1"

    input_dict = {
        "input": input_tensor,
        "depth_radius": depth_radius,
        "bias": bias,
        "alpha": alpha,
        "beta": beta,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.random.rand(2, 10, 10, 5).astype(np.float32)
    depth_radius = 5
    bias = 2.0
    alpha = 0.5
    beta = 0.2
    name = "lrn_2"

    input_dict = {
        "input": input_tensor,
        "depth_radius": depth_radius,
        "bias": bias,
        "alpha": alpha,
        "beta": beta,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.random.rand(1, 3, 3, 1).astype(np.float32)
    depth_radius = 1
    bias = 0.5
    alpha = 2.0
    beta = 1.0
    name = "lrn_3"

    input_dict = {
        "input": input_tensor,
        "depth_radius": depth_radius,
        "bias": bias,
        "alpha": alpha,
        "beta": beta,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.random.rand(4, 8, 8, 7).astype(np.float32)
    depth_radius = 3
    bias = 1.5
    alpha = 0.8
    beta = 0.7
    name = "lrn_4"

    input_dict = {
        "input": input_tensor,
        "depth_radius": depth_radius,
        "bias": bias,
        "alpha": alpha,
        "beta": beta,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 5
    input_tensor = np.random.rand(1, 5, 5, 3).astype(np.float32)
    depth_radius = 2
    bias = 1.0
    alpha = 1.0
    beta = 0.5
    name = "lrn_5"

    input_dict = {
        "input": input_tensor,
        "depth_radius": depth_radius,
        "bias": bias,
        "alpha": alpha,
        "beta": beta,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.random.rand(1, 5, 5, 3).astype(np.float32)
    depth_radius = 2
    bias = 1.0
    alpha = 1.0
    beta = 0.5
    name = "lrn_6"

    input_dict = {
        "input": input_tensor,
        "depth_radius": depth_radius,
        "bias": bias,
        "alpha": alpha,
        "beta": beta,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger depth_radius
    input_tensor = np.random.rand(1, 5, 5, 3).astype(np.float32)
    depth_radius = 4
    bias = 1.0
    alpha = 1.0
    beta = 0.5
    name = "lrn_7"

    input_dict = {
        "input": input_tensor,
        "depth_radius": depth_radius,
        "bias": bias,
        "alpha": alpha,
        "beta": beta,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.random.rand(1, 2, 2, 1).astype(np.float32)
    depth_radius = 0
    bias = 1.0
    alpha = 1.0
    beta = 0.5
    name = "lrn_8"

    input_dict = {
        "input": input_tensor,
        "depth_radius": depth_radius,
        "bias": bias,
        "alpha": alpha,
        "beta": beta,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.random.rand(1, 4, 4, 3).astype(np.float32)
    depth_radius = 1
    bias = 0.1
    alpha = 0.1
    beta = 0.1
    name = "lrn_9"

    input_dict = {
        "input": input_tensor,
        "depth_radius": depth_radius,
        "bias": bias,
        "alpha": alpha,
        "beta": beta,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.random.rand(2, 5, 5, 2).astype(np.float32)
    depth_radius = 2
    bias = 2.0
    alpha = 2.0
    beta = 2.0
    name = "lrn_10"

    input_dict = {
        "input": input_tensor,
        "depth_radius": depth_radius,
        "bias": bias,
        "alpha": alpha,
        "beta": beta,
        "name": name
    }
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
