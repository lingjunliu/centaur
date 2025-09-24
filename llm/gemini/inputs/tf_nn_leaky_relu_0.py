
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_leaky_relu_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 tensor
    features = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    alpha = 0.2
    name = "leaky_relu_1"
    input_dict = {"features": features, "alpha": alpha, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64 tensor with negative alpha
    features = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    alpha = -0.1
    name = "leaky_relu_2"
    input_dict = {"features": features, "alpha": alpha, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int32 tensor
    features = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.int32)
    alpha = 0.3
    name = "leaky_relu_3"
    input_dict = {"features": features, "alpha": alpha, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int64 tensor with alpha=0
    features = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4], dtype=np.int64)
    alpha = 0.0
    name = "leaky_relu_4"
    input_dict = {"features": features, "alpha": alpha, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D float32 tensor
    features = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.5, 2.0]], dtype=np.float32)
    alpha = 0.25
    name = "leaky_relu_5"
    input_dict = {"features": features, "alpha": alpha, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float16 tensor
    features = np.array([[[1.0, -1.0], [2.0, -2.0]], [[3.0, -3.0], [4.0, -4.0]]], dtype=np.float16)
    alpha = 0.4
    name = "leaky_relu_6"
    input_dict = {"features": features, "alpha": alpha, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32 tensor with large values
    features = np.array([-100.0, 0.0, 100.0], dtype=np.float32)
    alpha = 0.1
    name = "leaky_relu_7"
    input_dict = {"features": features, "alpha": alpha, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float32 tensor with small values
    features = np.array([-0.001, 0.0, 0.001], dtype=np.float32)
    alpha = 0.05
    name = "leaky_relu_8"
    input_dict = {"features": features, "alpha": alpha, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32 tensor with all negative values
    features = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    alpha = 0.35
    name = "leaky_relu_9"
    input_dict = {"features": features, "alpha": alpha, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 10: float32 tensor with all positive values
    features = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    alpha = 0.15
    name = "leaky_relu_10"
    input_dict = {"features": features, "alpha": alpha, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.leaky_relu"] = tf_nn_leaky_relu_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.leaky_relu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.leaky_relu'.")

check_valid('tf.nn.leaky_relu', generated_inputs['tf.nn.leaky_relu'], lib="tf", suffix=0)
