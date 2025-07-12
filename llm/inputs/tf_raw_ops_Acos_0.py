
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Acos_inputs():
    list_of_inputs = []

    # Input 1: Valid, float32, 1D
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Valid, float64, 2D
    x = np.array([[-1.0, 0.0], [0.5, 1.0]], dtype=np.float64)
    input_dict = {"x": x, "name": "acos_op"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Valid, bfloat16, 1D
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float16)
    x = tf.dtypes.cast(x, tf.bfloat16).numpy()
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Valid, half, 2D
    x = np.array([[-1.0, 0.0], [0.5, 1.0]], dtype=np.float16)
    input_dict = {"x": x, "name": "another_acos"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Valid, complex64, 1D
    x = np.array([-1.0 + 0j, 0.0 + 0j, 1.0 + 0j], dtype=np.complex64)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Valid, complex128, 2D
    x = np.array([[-1.0 + 0j, 0.0 + 0j], [0.5 + 0j, 1.0 + 0j]], dtype=np.complex128)
    input_dict = {"x": x, "name": "complex_acos"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Valid, float32, 3D
    x = np.array([[[ -1.0, -0.5], [0.0, 0.5]], [[0.5, 1.0], [-1.0, 0.0]]], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Valid, float64, 1D, values within range
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float64)
    input_dict = {"x": x, "name": "acos_with_outside_range"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Valid, half, 1D, all values are 0
    x = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float16)
    input_dict = {"x": x, "name": "zeros_acos"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Valid, complex128, 1D, values with imaginary components
    x = np.array([-0.5 + 0j, 0.0 + 0j, 0.5 + 0j], dtype=np.complex128)
    input_dict = {"x": x, "name": "complex_with_imaginary"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Acos"] = tf_raw_ops_Acos_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Acos' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Acos'.")

check_valid('tf.raw_ops.Acos', generated_inputs['tf.raw_ops.Acos'], lib="tf", suffix=0)
