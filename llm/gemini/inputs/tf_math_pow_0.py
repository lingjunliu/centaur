
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_pow_inputs():
    list_of_inputs = []

    # Input 1: Basic integer powers
    x = np.array([[2, 3], [4, 5]], dtype=np.int32)
    y = np.array([[2, 3], [1, 0]], dtype=np.int32)
    name = "int_powers"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Floating point powers
    x = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    y = np.array([[2.0, 0.5], [1.5, 1.0]], dtype=np.float32)
    name = "float_powers"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Mixed integer and floating point (but both should be float to work generally)
    x = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    y = np.array([[2.0, 0.5], [1.5, 1.0]], dtype=np.float32)
    name = "mixed_powers"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative powers
    x = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    y = np.array([[-2.0, -0.5], [-1.5, -1.0]], dtype=np.float32)
    name = "negative_powers"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multi-dimensional tensors
    x = np.array([[[2.0, 3.0], [4.0, 5.0]], [[6.0, 7.0], [8.0, 9.0]]], dtype=np.float32)
    y = np.array([[[2.0, 0.5], [1.5, 1.0]], [[0.0, -0.5], [1.0, 2.0]]], dtype=np.float32)
    name = "multi_dim_powers"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Integer tensor with large values
    x = np.array([[2, 3], [4, 5]], dtype=np.int64)
    y = np.array([[2, 3], [1, 0]], dtype=np.int64)
    name = "int64_powers"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Float tensor with large values
    x = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float64)
    y = np.array([[2.0, 0.5], [1.5, 1.0]], dtype=np.float64)
    name = "float64_powers"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Complex numbers
    x = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex64)
    y = np.array([[1, 2], [0.5, 1]], dtype=np.complex64) # Keep y as complex
    name = "complex64_powers"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9: Zero values
    x = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    y = np.array([[2.0, 0.5], [1.5, 1.0]], dtype=np.float32)
    name = "zero_values"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: One value
    x = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    y = np.array([[2.0, 0.5], [1.5, 1.0]], dtype=np.float32)
    name = "one_value"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Different shaped tensors where broadcasting is applicable
    x = np.array([[2.0, 3.0]], dtype=np.float32)
    y = np.array([[2.0], [0.5]], dtype=np.float32)
    name = "broadcast_powers"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.pow"] = tf_math_pow_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.pow' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.pow'.")

check_valid('tf.math.pow', generated_inputs['tf.math.pow'], lib="tf", suffix=0)
