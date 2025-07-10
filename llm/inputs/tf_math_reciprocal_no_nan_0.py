
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_reciprocal_no_nan_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 tensor with 0 and non-zero values
    x = tf.constant([2.0, 0.5, 0.0, 1.0], dtype=tf.float32).numpy()
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64 tensor with negative values and 0
    x = tf.constant([-2.0, -0.5, 0.0, 1.0], dtype=tf.float64).numpy()
    name = "reciprocal_op"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float16 tensor
    x = tf.constant([2.0, 0.5, 0.0, 1.0], dtype=tf.float16).numpy()
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Complex64 tensor
    x = tf.constant([1+1j, 0+0j, 2-2j], dtype=tf.complex64).numpy()
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex128 tensor
    x = tf.constant([1+1j, 0+0j, 2-2j], dtype=tf.complex128).numpy()
    name = "complex_reciprocal"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D float32 tensor
    x = tf.constant([[1.0, 2.0], [0.0, 4.0]], dtype=tf.float32).numpy()
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D float64 tensor
    x = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 0.0], [7.0, 8.0]]], dtype=tf.float64).numpy()
    name = "3d_reciprocal"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: All zeros float32 tensor
    x = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32).numpy()
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32 tensor with nan values
    x = tf.constant([np.nan, 1.0, 0.0], dtype=tf.float32).numpy()
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32 scalar tensor
    x = tf.constant(5.0, dtype=tf.float32).numpy()
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.reciprocal_no_nan"] = tf_math_reciprocal_no_nan_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.reciprocal_no_nan' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.reciprocal_no_nan'.")

check_valid('tf.math.reciprocal_no_nan', generated_inputs['tf.math.reciprocal_no_nan'], lib="tf", suffix=0)
