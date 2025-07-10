
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_squared_difference_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 tensors
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    name = "basic_float32"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different shapes, broadcasting
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([1.0, 2.0], dtype=np.float32)
    name = "broadcasting"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative values
    x = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    y = np.array([-4.0, -5.0, -6.0], dtype=np.float32)
    name = "negative_values"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Integer values
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([4, 5, 6], dtype=np.int32)
    name = "integer_values"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex values
    x = np.array([1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex64)
    y = np.array([4 + 4j, 5 + 5j, 6 + 6j], dtype=np.complex64)
    name = "complex_values"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty arrays
    x = np.array([], dtype=np.float32)
    y = np.array([], dtype=np.float32)
    name = "empty_arrays"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Multi-dimensional array
    x = np.array([[1, 2], [3, 4]], dtype=np.int64)
    y = np.array([[5, 6], [7, 8]], dtype=np.int64)
    name = "multi_dimensional"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float16
    x = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    y = np.array([4.0, 5.0, 6.0], dtype=np.float16)
    name = "float16"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: half
    x = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    y = np.array([4.0, 5.0, 6.0], dtype=np.float16)
    name = "half"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float64
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    name = "float64"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: complex128
    x = np.array([1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex128)
    y = np.array([4 + 4j, 5 + 5j, 6 + 6j], dtype=np.complex128)
    name = "complex128"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.squared_difference"] = tf_math_squared_difference_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.squared_difference' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.squared_difference'.")

check_valid('tf.math.squared_difference', generated_inputs['tf.math.squared_difference'], lib="tf", suffix=0)
