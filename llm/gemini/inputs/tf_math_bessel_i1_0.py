
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_bessel_i1_inputs():
    list_of_inputs = []

    # Input 1: Simple float32 tensor
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative float32 tensor
    x = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    name = "negative_values"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64 tensor
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: half tensor
    x = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: zero values
    x = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    name = "zero_values"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: large values
    x = np.array([100.0, 200.0, 300.0], dtype=np.float32)
    name = "large_values"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D tensor
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D tensor
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    name = "3d_tensor"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Mixed positive and negative values
    x = np.array([-1.0, 0.5, 2.0, -3.5], dtype=np.float32)
    name = "mixed_values"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Tensor with small values
    x = np.array([0.1, 0.01, 0.001], dtype=np.float32)
    name = "small_values"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.bessel_i1"] = tf_math_bessel_i1_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.bessel_i1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.bessel_i1'.")

check_valid('tf.math.bessel_i1', generated_inputs['tf.math.bessel_i1'], lib="tf", suffix=0)
