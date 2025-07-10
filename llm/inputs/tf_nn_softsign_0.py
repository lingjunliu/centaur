
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_softsign_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 tensor
    features = np.array([1.0, 2.0, -1.0, -2.0], dtype=np.float32)
    name = "softsign_1"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float64 tensor with zeros
    features = np.array([0.0, 1.0, -1.0, 0.0], dtype=np.float64)
    name = "softsign_2"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multi-dimensional float32 tensor
    features = np.array([[1.0, 2.0], [-1.0, -2.0]], dtype=np.float32)
    name = "softsign_3"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: bfloat16 tensor
    features = np.array([0.5, -0.5, 1.5, -1.5], dtype=np.float32).astype(np.float16)
    name = "softsign_4"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large values in float32 tensor
    features = np.array([100.0, -100.0, 1000.0, -1000.0], dtype=np.float32)
    name = "softsign_5"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: half tensor (float16)
    features = np.array([0.25, -0.25, 0.75, -0.75], dtype=np.float32).astype(np.float16)
    name = "softsign_6"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D tensor with float64
    features = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64)
    name = "softsign_7"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8: Tensor with a mix of positive and negative values and zeros
    features = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    name = "softsign_8"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Tensor with very small values
    features = np.array([0.0001, -0.0001, 0.001, -0.001], dtype=np.float32)
    name = "softsign_9"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: empty array
    features = np.array([], dtype=np.float32)
    name = "softsign_10"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: float64 array
    features = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float64)
    name = "softsign_11"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.softsign"] = tf_nn_softsign_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.softsign' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.softsign'.")

check_valid('tf.nn.softsign', generated_inputs['tf.nn.softsign'], lib="tf", suffix=0)
