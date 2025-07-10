
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_divide_inputs():
    list_of_inputs = []

    # Input 1: Basic integer division
    x = tf.constant(np.array([10, 20, 30]), dtype=tf.int32).numpy()
    y = tf.constant(np.array([2, 5, 10]), dtype=tf.int32).numpy()
    name = "basic_division"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Floating-point division
    x = tf.constant(np.array([1.0, 2.5, 3.7]), dtype=tf.float32).numpy()
    y = tf.constant(np.array([0.5, 1.0, 2.0]), dtype=tf.float32).numpy()
    name = "float_division"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Division with negative numbers
    x = tf.constant(np.array([-10, 20, -30]), dtype=tf.int32).numpy()
    y = tf.constant(np.array([2, -5, 10]), dtype=tf.int32).numpy()
    name = "negative_division"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Division by zero (should not crash, but produce inf or nan)
    x = tf.constant(np.array([1, 2, 3]), dtype=tf.float32).numpy()
    y = tf.constant(np.array([0.0, 0.0, 0.0]), dtype=tf.float32).numpy()
    name = "division_by_zero"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D Tensor division
    x = tf.constant(np.array([[1, 2], [3, 4]]), dtype=tf.float32).numpy()
    y = tf.constant(np.array([[2, 1], [4, 3]]), dtype=tf.float32).numpy()
    name = "2d_division"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D Tensor division
    x = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), dtype=tf.float32).numpy()
    y = tf.constant(np.array([[[2, 1], [4, 3]], [[6, 5], [8, 7]]]), dtype=tf.float32).numpy()
    name = "3d_division"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7: Different dtypes (int64)
    x = tf.constant(np.array([10, 20, 30]), dtype=tf.int64).numpy()
    y = tf.constant(np.array([2, 5, 10]), dtype=tf.int64).numpy()
    name = "int64_division"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different dtypes (float64)
    x = tf.constant(np.array([1.0, 2.5, 3.7]), dtype=tf.float64).numpy()
    y = tf.constant(np.array([0.5, 1.0, 2.0]), dtype=tf.float64).numpy()
    name = "float64_division"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Scalar division
    x = tf.constant(10, dtype=tf.int32).numpy()
    y = tf.constant(2, dtype=tf.int32).numpy()
    name = "scalar_division"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Tensor with zeros
    x = tf.constant(np.array([0, 20, 0]), dtype=tf.int32).numpy()
    y = tf.constant(np.array([2, 5, 10]), dtype=tf.int32).numpy()
    name = "zero_values"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.divide"] = tf_math_divide_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.divide' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.divide'.")

check_valid('tf.math.divide', generated_inputs['tf.math.divide'], lib="tf", suffix=0)
