
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_IsFinite_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 array
    x = np.array([1.0, 2.0, 3.0, np.inf, np.nan], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(input_dict)

    # Input 2: float64 array with negative values
    x = np.array([-1.0, -2.0, 0.0, np.inf, np.nan, -np.inf], dtype=np.float64)
    input_dict = {"x": x, "name": "negative_values"}
    list_of_inputs.append(input_dict)

    # Input 3: half (float16) array
    x = np.array([1.0, 2.0, 3.0, np.inf, np.nan], dtype=np.float16)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(input_dict)

    # Input 4: bfloat16 array
    x = tf.constant([1.0, 2.0, 3.0, np.inf, np.nan], dtype=tf.bfloat16).numpy()
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(input_dict)

    # Input 5: Multi-dimensional float32 array
    x = np.array([[1.0, 2.0], [np.inf, np.nan]], dtype=np.float32)
    input_dict = {"x": x, "name": "multi_dim"}
    list_of_inputs.append(input_dict)

    # Input 6: float32 array with a name
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"x": x, "name": "valid_floats"}
    list_of_inputs.append(input_dict)

    # Input 7: float64 array with very large numbers
    x = np.array([1e10, 2e20, 3e30, np.inf], dtype=np.float64)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(input_dict)
    
    # Input 8: float32 zero value
    x = np.array([0.0], dtype=np.float32)
    input_dict = {"x": x, "name": "zero_value"}
    list_of_inputs.append(input_dict)

    # Input 9: float32 array with a mix of finite and infinite values
    x = np.array([1.0, np.inf, 2.0, -np.inf, 3.0, np.nan], dtype=np.float32)
    input_dict = {"x": x, "name": "mixed_values"}
    list_of_inputs.append(input_dict)

    # Input 10: float64 array with small numbers
    x = np.array([1e-10, 2e-20, 3e-30], dtype=np.float64)
    input_dict = {"x": x, "name": "small_numbers"}
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.IsFinite"] = tf_raw_ops_IsFinite_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.IsFinite' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.IsFinite'.")

check_valid('tf.raw_ops.IsFinite', generated_inputs['tf.raw_ops.IsFinite'], lib="tf", suffix=0)
