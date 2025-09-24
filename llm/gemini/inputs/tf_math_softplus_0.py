
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_softplus_inputs():
    list_of_inputs = []

    # Input 1: Scalar input
    features = tf.constant(0.0, dtype=tf.float32).numpy()
    name = "scalar_input"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D Tensor with positive values
    features = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32).numpy()
    name = "positive_1d"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D Tensor with negative values
    features = tf.constant([-1.0, -2.0, -3.0], dtype=tf.float32).numpy()
    name = "negative_1d"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D Tensor with mixed values
    features = tf.constant([-1.0, 0.0, 1.0], dtype=tf.float32).numpy()
    name = "mixed_1d"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D Tensor
    features = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32).numpy()
    name = "2d_tensor"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D Tensor with negative values
    features = tf.constant([[-1.0, -2.0], [-3.0, -4.0]], dtype=tf.float32).numpy()
    name = "2d_tensor_negative"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D Tensor
    features = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=tf.float32).numpy()
    name = "3d_tensor"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Tensor with dtype float64
    features = tf.constant([1.0, 2.0, 3.0], dtype=tf.float64).numpy()
    name = "float64_tensor"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Tensor with dtype float16
    features = tf.constant([1.0, 2.0, 3.0], dtype=tf.float16).numpy()
    name = "float16_tensor"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Large values
    features = tf.constant([100.0, 1000.0, 10000.0], dtype=tf.float32).numpy()
    name = "large_values"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.softplus"] = tf_math_softplus_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.softplus' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.softplus'.")

check_valid('tf.math.softplus', generated_inputs['tf.math.softplus'], lib="tf", suffix=0)
