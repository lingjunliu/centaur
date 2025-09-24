
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_special_spence_inputs():
    list_of_inputs = []

    # Input 1: Scalar float32
    x = tf.constant(0.5, dtype=tf.float32)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Scalar float64
    x = tf.constant(1.0, dtype=tf.float64)
    name = "spence_integral"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D float32 array
    x = tf.constant([0.5, 1.0, 2.0, 3.0], dtype=tf.float32)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D float64 array with negative values
    x = tf.constant([-0.5, 0.0, 1.5, 2.5], dtype=tf.float64)
    name = "negative_example"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D float32 array
    x = tf.constant([[0.2, 0.4], [0.6, 0.8]], dtype=tf.float32)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D float64 array with varied values
    x = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float64)
    name = "2d_array"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D float32 array
    x = tf.constant([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=tf.float32)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D float64 array with negative values
    x = tf.constant([[[(-0.1), 0.2], [0.3, (-0.4)]], [[0.5, (-0.6)], [(-0.7), 0.8]]], dtype=tf.float64)
    name = "3d_negative"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: float32 with large values
    x = tf.constant([100.0, 200.0], dtype=tf.float32)
    name = "large_values"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float64 array with very small values.
    x = tf.constant([0.0001, 0.0002], dtype=tf.float64)
    name = "small_values"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.special.spence"] = tf_math_special_spence_inputs()
for i in range(len(generated_inputs["tf.math.special.spence"])):
    generated_inputs["tf.math.special.spence"][i]["x"] = generated_inputs["tf.math.special.spence"][i]["x"].numpy()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.special.spence' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.special.spence'.")

check_valid('tf.math.special.spence', generated_inputs['tf.math.special.spence'], lib="tf", suffix=0)
