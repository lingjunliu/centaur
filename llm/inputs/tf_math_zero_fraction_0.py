
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_zero_fraction_inputs():
    list_of_inputs = []

    # Input 1: All zeros
    value = np.zeros((5, 5), dtype=np.float32)
    name = "all_zeros"
    input_dict = {"value": value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: All non-zeros
    value = np.ones((3, 3), dtype=np.float32)
    name = "all_ones"
    input_dict = {"value": value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Mixed zeros and non-zeros
    value = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
    name = "mixed"
    input_dict = {"value": value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative and positive values with zeros
    value = np.array([[-1, 0, 2], [0, -3, 0], [4, 0, -5]], dtype=np.float32)
    name = "negative_positive"
    input_dict = {"value": value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: A 1D array
    value = np.array([0, 1, 2, 0, 3, 0], dtype=np.float32)
    name = "1d_array"
    input_dict = {"value": value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: A 3D array
    value = np.array([[[0, 1], [2, 0]], [[3, 0], [4, 5]]], dtype=np.float32)
    name = "3d_array"
    input_dict = {"value": value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: A large array with some zeros
    value = np.random.rand(100, 100).astype(np.float32)
    value[value < 0.1] = 0
    name = "large_array"
    input_dict = {"value": value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Only one element which is zero
    value = np.array([0], dtype=np.float32)
    name = "single_zero"
    input_dict = {"value": value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Only one element which is non-zero
    value = np.array([1], dtype=np.float32)
    name = "single_non_zero"
    input_dict = {"value": value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Array of integers
    value = np.array([[0, 1, 0], [2, 0, 3]], dtype=np.int32)
    name = "int_array"
    input_dict = {"value": value.astype(np.float32), "name": name} # Convert to float32 as expected by tf.math.zero_fraction
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.zero_fraction"] = tf_math_zero_fraction_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.zero_fraction' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.zero_fraction'.")

check_valid('tf.math.zero_fraction', generated_inputs['tf.math.zero_fraction'], lib="tf", suffix=0)
