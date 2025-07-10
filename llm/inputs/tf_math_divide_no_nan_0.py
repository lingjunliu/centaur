
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_divide_no_nan_inputs():
    list_of_inputs = []

    # Input 1: Basic float division, no NaN
    x = tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float32)).numpy()
    y = tf.constant(np.array([4.0, 5.0, 6.0], dtype=np.float32)).numpy()
    name = None
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Division by zero
    x = tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float32)).numpy()
    y = tf.constant(np.array([0.0, 0.0, 0.0], dtype=np.float32)).numpy()
    name = "divide_by_zero"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Mixed positive and negative, with a zero
    x = tf.constant(np.array([-1.0, 2.0, 0.0], dtype=np.float32)).numpy()
    y = tf.constant(np.array([1.0, -2.0, 0.0], dtype=np.float32)).numpy()
    name = "mixed_values"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Integer division by zero
    x = tf.constant(np.array([1, 2, 3], dtype=np.int32)).numpy()
    y = tf.constant(np.array([0, 0, 0], dtype=np.int32)).numpy()
    name = "integer_division"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D tensor
    x = tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)).numpy()
    y = tf.constant(np.array([[5.0, 0.0], [7.0, 8.0]], dtype=np.float32)).numpy()
    name = "2d_tensor"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different shape but broadcastable
    x = tf.constant(np.array([1.0, 2.0], dtype=np.float32)).numpy()
    y = tf.constant(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)).numpy()
    name = "broadcast"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7: Large values
    x = tf.constant(np.array([1e9, 2e9], dtype=np.float32)).numpy()
    y = tf.constant(np.array([1.0, 0.0], dtype=np.float32)).numpy()
    name = "large_values"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small values
    x = tf.constant(np.array([1e-9, 2e-9], dtype=np.float32)).numpy()
    y = tf.constant(np.array([1.0, 0.0], dtype=np.float32)).numpy()
    name = "small_values"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D tensor
    x = tf.constant(np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)).numpy()
    y = tf.constant(np.array([[[5.0, 0.0], [7.0, 8.0]], [[1.0, 2.0], [0.0, 1.0]]], dtype=np.float32)).numpy()
    name = "3d_tensor"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: All zeros
    x = tf.constant(np.array([0.0, 0.0, 0.0], dtype=np.float32)).numpy()
    y = tf.constant(np.array([0.0, 0.0, 0.0], dtype=np.float32)).numpy()
    name = "all_zeros"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.divide_no_nan"] = tf_math_divide_no_nan_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.divide_no_nan' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.divide_no_nan'.")

check_valid('tf.math.divide_no_nan', generated_inputs['tf.math.divide_no_nan'], lib="tf", suffix=0)
