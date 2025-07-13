
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_relu6_inputs():
    list_of_inputs = []

    # Input 1: float32, 1D array
    features = np.array([-1.0, 0.0, 1.0, 5.0, 7.0], dtype=np.float32)
    name = "relu6_float32_1d"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, 2D array
    features = np.array([[-2.0, 0.5, 2.0], [4.0, 6.5, 8.0]], dtype=np.float64)
    name = "relu6_float64_2d"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int32, 3D array
    features = np.array([[[1, -1], [2, 0]], [[3, 5], [4, 7]]], dtype=np.int32)
    name = "relu6_int32_3d"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: uint8, scalar
    features = np.array(3, dtype=np.uint8)
    name = "relu6_uint8_scalar"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: int16, 1D array with large values
    features = np.array([-100, 200, 5, 7000], dtype=np.int16)
    name = "relu6_int16_large"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_raw_ops_relu6_inputs()
generated_inputs["tf.raw_ops.Relu6"] = []
for input_dict in inputs:
  generated_inputs["tf.raw_ops.Relu6"].append(input_dict)

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Relu6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Relu6'.")

check_valid('tf.raw_ops.Relu6', generated_inputs['tf.raw_ops.Relu6'], lib="tf", suffix=0)
