
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_multiply_no_nan_inputs():
    list_of_inputs = []

    # Input 1: Basic case with positive numbers
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([4.0, 0.0, 6.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "basic_positive"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative numbers
    x = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    y = np.array([4.0, 0.0, -6.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "negative_numbers"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Zeros
    x = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    y = np.array([4.0, 0.0, -6.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "zeros"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different shapes (broadcasting)
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([0.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "broadcasting"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: NaN in x, zero in y
    x = np.array([np.nan, 2.0, 3.0], dtype=np.float32)
    y = np.array([0.0, 0.0, 6.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "nan_x_zero_y"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Inf in x, zero in y
    x = np.array([np.inf, 2.0, 3.0], dtype=np.float32)
    y = np.array([0.0, 0.0, 6.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "inf_x_zero_y"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: NaN in y
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([np.nan, 0.0, 6.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "nan_y"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger arrays
    x = np.random.rand(100, 100).astype(np.float32)
    y = np.random.rand(100, 100).astype(np.float32)
    y[50, 50] = 0.0
    input_dict = {"x": x, "y": y, "name": "larger_arrays"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Double precision
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y = np.array([4.0, 0.0, 6.0], dtype=np.float64)
    input_dict = {"x": x, "y": y, "name": "double_precision"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: x is zero, y is nan
    x = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    y = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "zero_x_nan_y"}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.multiply_no_nan"] = tf_math_multiply_no_nan_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.multiply_no_nan' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.multiply_no_nan'.")

check_valid('tf.math.multiply_no_nan', generated_inputs['tf.math.multiply_no_nan'], lib="tf", suffix=0)
