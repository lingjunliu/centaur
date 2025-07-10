
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_is_strictly_increasing_inputs():
    list_of_inputs = []

    # Input 1: Strictly increasing, 1D
    x = tf.constant(np.array([1, 2, 3, 4, 5], dtype=np.float32))
    name = "strictly_increasing_1d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Not strictly increasing, 1D
    x = tf.constant(np.array([1, 2, 3, 3, 5], dtype=np.float32))
    name = "not_strictly_increasing_1d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Strictly increasing, 2D
    x = tf.constant(np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32))
    name = "strictly_increasing_2d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Not strictly increasing, 2D
    x = tf.constant(np.array([[1, 2], [3, 4], [5, 5]], dtype=np.int32))
    name = "not_strictly_increasing_2d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Strictly increasing, 3D
    x = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float64))
    name = "strictly_increasing_3d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Not strictly increasing, 3D
    x = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 7]]], dtype=np.float64))
    name = "not_strictly_increasing_3d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Negative values, strictly increasing
    x = tf.constant(np.array([-5, -4, -3, -2, -1], dtype=np.int64))
    name = "negative_strictly_increasing"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Mixed positive and negative, strictly increasing
    x = tf.constant(np.array([-1, 0, 1, 2, 3], dtype=np.int32))
    name = "mixed_strictly_increasing"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Tensor with zero
    x = tf.constant(np.array([0, 1, 2, 3, 4], dtype=np.float32))
    name = "zero_strictly_increasing"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Empty tensor, should return True
    x = tf.constant(np.array([], dtype=np.float32))
    name = "empty_tensor"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Single element tensor, should return True
    x = tf.constant(np.array([5], dtype=np.float32))
    name = "single_element_tensor"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: Large values, strictly increasing
    x = tf.constant(np.array([1e9, 1e9 + 1, 1e9 + 2], dtype=np.float32))
    name = "large_values_strictly_increasing"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 13: Small values, strictly increasing
    x = tf.constant(np.array([1e-9, 1e-8, 1e-7], dtype=np.float32))
    name = "small_values_strictly_increasing"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 14: Mixed small and large values, not strictly increasing
    x = tf.constant(np.array([1e-9, 1e-8, 1e-7, 0.0000001, 1e9, 1e9 + 1, 1e9 + 2, 1e9], dtype=np.float32))
    name = "mixed_small_large_not_strictly_increasing"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.is_strictly_increasing"] = tf_math_is_strictly_increasing_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.is_strictly_increasing' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.is_strictly_increasing'.")

check_valid('tf.math.is_strictly_increasing', generated_inputs['tf.math.is_strictly_increasing'], lib="tf", suffix=0)
