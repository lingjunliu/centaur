
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_rint_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 scalar
    x = np.float32(2.3)
    name = "scalar_float32"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic float64 scalar
    x = np.float64(-1.7)
    name = "scalar_float64"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32 array with positive and negative values
    x = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], dtype=np.float32)
    name = "float32_array"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64 array with values near integers
    x = np.array([-1.000001, 0.999999, 2.000001, -2.999999], dtype=np.float64)
    name = "float64_near_integers"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32 2D array
    x = np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float32)
    name = "float32_2d_array"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float64 2D array with negative values
    x = np.array([[-1.1, -2.2], [-3.3, -4.4]], dtype=np.float64)
    name = "float64_2d_array_negative"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32 3D array
    x = np.array([[[1.1, 2.2], [3.3, 4.4]], [[5.5, 6.6], [7.7, 8.8]]], dtype=np.float32)
    name = "float32_3d_array"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float64 3D array with 0
    x = np.array([[[1.0, 2.5], [3.0, 4.5]], [[5.0, 6.5], [7.0, 0.0]]], dtype=np.float64)
    name = "float64_3d_array_with_zero"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: half array
    x = np.array([1.1, 2.2, 3.3, 4.4], dtype=np.float16)
    name = "half_array"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32 with large values
    x = np.array([1000000.1, -1000000.2], dtype=np.float32)
    name = "float32_large_values"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.rint"] = tf_math_rint_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.rint' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.rint'.")

check_valid('tf.math.rint', generated_inputs['tf.math.rint'], lib="tf", suffix=0)
