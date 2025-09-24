
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_igammac_inputs():
    list_of_inputs = []

    # Input 1: Basic case with float32
    a = np.array([2.0], dtype=np.float32)
    x = np.array([3.0], dtype=np.float32)
    name = "igammac_basic_float32"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, multi-dimensional
    a = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
    x = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float64)
    name = "igammac_multi_float64"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Higher dimensional tensor, float32
    a = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    x = np.array([[[2.0, 3.0], [4.0, 5.0]], [[6.0, 7.0], [8.0, 9.0]]], dtype=np.float32)
    name = "igammac_high_dim_float32"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: with different values for a and x, float32
    a = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    name = "igammac_diff_values_float32"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32, different shape
    a = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    x = np.array([[2.0], [3.0], [4.0]], dtype=np.float32)
    name = "igammac_diff_shape_float32"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Single element tensors, float32
    a = np.array(1.0, dtype=np.float32)
    x = np.array(2.0, dtype=np.float32)
    name = "igammac_single_float32"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: float64 type
    a = np.array([2.718, 3.1415], dtype=np.float64)
    x = np.array([1.618, 2.718], dtype=np.float64)
    name = "igammac_float64_2"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: More complex float32
    a = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    x = np.array([[1.1, 1.2], [1.3, 1.4]], dtype=np.float32)
    name = "igammac_float32_complex"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Basic case with float32 and name
    a = np.array([0.5], dtype=np.float32)
    x = np.array([1.5], dtype=np.float32)
    name = "igammac_named_float32"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Single float64 value
    a = np.array(3.0, dtype=np.float64)
    x = np.array(4.0, dtype=np.float64)
    name = "igammac_single_float64"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.igammac"] = tf_math_igammac_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.igammac' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.igammac'.")

check_valid('tf.math.igammac', generated_inputs['tf.math.igammac'], lib="tf", suffix=0)
