
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Erfinv_inputs():
    list_of_inputs = []

    # Input 1: float32, 1D
    x = np.array([0.0, 0.5, 0.9], dtype=np.float32)
    name = "erfinv_1"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, 2D
    x = np.array([[-0.2, 0.3], [0.7, -0.9]], dtype=np.float64)
    name = "erfinv_2"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float16, 1D
    x = np.array([-0.8, 0.1, 0.6], dtype=np.float16)
    name = "erfinv_3"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float16, 2D
    x = np.array([[0.4, -0.5], [-0.1, 0.8]], dtype=np.float16)
    name = "erfinv_4"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32, 3D
    x = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float32)
    name = "erfinv_5"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float64, 1D with zeros
    x = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    name = "erfinv_6"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32, with negative and positive values close to 1
    x = np.array([-0.9, 0.9, -0.01, 0.01], dtype=np.float32)
    name = "erfinv_7"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float16, larger dimension
    x = np.array([[-0.8, 0.7, -0.6], [0.5, -0.4, 0.3], [-0.2, 0.1, -0.0]], dtype=np.float16)
    name = "erfinv_8"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float64, small values
    x = np.array([1e-5, -1e-5, 1e-8, -1e-8], dtype=np.float64)
    name = "erfinv_9"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32, array of ones
    x = np.ones((4,4), dtype=np.float32) * 0.5
    name = "erfinv_10"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Erfinv"] = tf_raw_ops_Erfinv_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Erfinv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Erfinv'.")

check_valid('tf.raw_ops.Erfinv', generated_inputs['tf.raw_ops.Erfinv'], lib="tf", suffix=0)
