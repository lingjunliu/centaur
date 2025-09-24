
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_lgamma_inputs():
    list_of_inputs = []

    # Input 1: Basic float32
    x = np.array([0.5, 1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float64 with negative values
    x = np.array([-0.5, -1.0, -2.0, -3.0], dtype=np.float64)
    input_dict = {"x": x, "name": "negative_values"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Half type
    x = np.array([0.5, 1.0, 2.0, 3.0], dtype=np.float16)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: bfloat16 type (converted to float32)
    x = np.array([0.5, 1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D float32
    x = np.array([[0.5, 1.0], [2.0, 3.0]], dtype=np.float32)
    input_dict = {"x": x, "name": "2d_array"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D Float64 with zeros
    x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    input_dict = {"x": x, "name": "zeros"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Float32 with large values
    x = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    input_dict = {"x": x, "name": "large_values"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Float64 with fractional and negative values
    x = np.array([-0.25, 1.75, -2.5, 3.125], dtype=np.float64)
    input_dict = {"x": x, "name": "fractional_and_negative"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 3D float32 array
    x = np.array([[[0.5, 1.0], [1.5, 2.0]], [[2.5, 3.0], [3.5, 4.0]]], dtype=np.float32)
    input_dict = {"x": x, "name": "3d_array"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10:  Float32 with negative and positive values
    x = np.array([-1.0, 0.5, 2.0, -3.0], dtype=np.float32)
    input_dict = {"x": x, "name": "float32_neg_pos"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Lgamma"] = tf_raw_ops_lgamma_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Lgamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Lgamma'.")

check_valid('tf.raw_ops.Lgamma', generated_inputs['tf.raw_ops.Lgamma'], lib="tf", suffix=0)
