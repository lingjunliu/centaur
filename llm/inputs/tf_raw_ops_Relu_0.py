
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_relu_inputs():
    list_of_inputs = []

    # Input 1: float32, 1D, positive values
    features = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"features": features, "name": "relu_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32, 1D, mixed positive and negative values
    features = np.array([-1.0, 0.0, 3.0, -5.0], dtype=np.float32)
    input_dict = {"features": features, "name": "relu_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32, 2D, positive values
    features = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"features": features, "name": "relu_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float32, 2D, mixed positive and negative values
    features = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    input_dict = {"features": features, "name": "relu_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: int32, 1D, positive values
    features = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {"features": features, "name": "relu_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: int32, 1D, mixed positive and negative values
    features = np.array([-1, 0, 3, -5], dtype=np.int32)
    input_dict = {"features": features, "name": "relu_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: int32, 2D, positive values
    features = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict = {"features": features, "name": "relu_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: int32, 2D, mixed positive and negative values
    features = np.array([[-1, 2], [3, -4]], dtype=np.int32)
    input_dict = {"features": features, "name": "relu_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float64, 3D, mixed positive and negative values
    features = np.array([[[1.0, -2.0], [3.0, -4.0]], [[-5.0, 6.0], [-7.0, 8.0]]], dtype=np.float64)
    input_dict = {"features": features, "name": "relu_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 10: uint8, 1D, values
    features = np.array([1, 2, 255, 0], dtype=np.uint8)
    input_dict = {"features": features, "name": "relu_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Relu"] = tf_raw_ops_relu_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Relu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Relu'.")

check_valid('tf.raw_ops.Relu', generated_inputs['tf.raw_ops.Relu'], lib="tf", suffix=0)
