
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_bessel_i1e_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 array
    x = np.array([-1.0, -0.5, 0.5, 1.0], dtype=np.float32)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64 array
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    name = "bessel_i1e_float64"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: half array (float16)
    x = np.array([-0.75, -0.25, 0.25, 0.75], dtype=np.float16)
    name = "bessel_i1e_half"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multi-dimensional float32 array
    x = np.array([[-1.0, 0.0], [0.5, 1.0]], dtype=np.float32)
    name = "bessel_i1e_2d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative values, float64
    x = np.array([-5.0, -2.5, -0.1], dtype=np.float64)
    name = "bessel_i1e_negative"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Zero values, float32
    x = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    name = "bessel_i1e_zero"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large values, float64
    x = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    name = "bessel_i1e_large"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D array, float32
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    name = "bessel_i1e_3d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9: Mixed positive and negative values, float16
    x = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float16)
    name = "bessel_i1e_mixed"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: name provided
    x = np.array([1.0, 2.0], dtype=np.float32)
    name = "my_bessel"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.bessel_i1e"] = tf_math_bessel_i1e_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.bessel_i1e' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.bessel_i1e'.")

check_valid('tf.math.bessel_i1e', generated_inputs['tf.math.bessel_i1e'], lib="tf", suffix=0)
