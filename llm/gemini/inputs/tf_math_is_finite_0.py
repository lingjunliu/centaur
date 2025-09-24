
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_is_finite_inputs():
    list_of_inputs = []

    # Input 1: Basic finite values
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: With negative values
    x = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    input_dict = {"x": x, "name": "negative_values"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: With zero
    x = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: With a mix of positive, negative and zero
    x = np.array([-1.0, 0.0, 2.0], dtype=np.float32)
    input_dict = {"x": x, "name": "mixed_values"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict = {"x": x, "name": "3d_array"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: bfloat16
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: half
    x = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    input_dict = {"x": x, "name": "half_precision"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float64
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Large values
    x = np.array([1e10, 2e10, 3e10], dtype=np.float32)
    input_dict = {"x": x, "name": "large_values"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.is_finite"] = tf_math_is_finite_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.is_finite' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.is_finite'.")

check_valid('tf.math.is_finite', generated_inputs['tf.math.is_finite'], lib="tf", suffix=0)
