
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_round_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 tensor
    x = tf.constant([0.9, 2.5, 2.3, 1.5, -4.5], dtype=tf.float32)
    name = None
    input_dict = {"x": x.numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64 tensor
    x = tf.constant([0.9, 2.5, 2.3, 1.5, -4.5], dtype=tf.float64)
    name = "float64_tensor"
    input_dict = {"x": x.numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int32 tensor
    x = tf.constant([1, 2, 3, 4, 5], dtype=tf.int32)
    name = "int32_tensor"
    input_dict = {"x": x.numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int64 tensor
    x = tf.constant([1, 2, 3, 4, 5], dtype=tf.int64)
    name = "int64_tensor"
    input_dict = {"x": x.numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative float32 tensor
    x = tf.constant([-0.9, -2.5, -2.3, -1.5, -4.5], dtype=tf.float32)
    name = "negative_float32"
    input_dict = {"x": x.numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D float32 tensor
    x = tf.constant([[0.9, 2.5], [2.3, 1.5]], dtype=tf.float32)
    name = "2d_float32"
    input_dict = {"x": x.numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D float32 tensor
    x = tf.constant([[[0.9, 2.5], [2.3, 1.5]], [[-0.9, -2.5], [-2.3, -1.5]]], dtype=tf.float32)
    name = "3d_float32"
    input_dict = {"x": x.numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float16 tensor
    x = tf.constant([0.9, 2.5, 2.3, 1.5, -4.5], dtype=tf.float16)
    name = "float16_tensor"
    input_dict = {"x": x.numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: large values
    x = tf.constant([1000.9, 2000.5, -3000.3], dtype=tf.float32)
    name = "large_values"
    input_dict = {"x": x.numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Tensor with zero values
    x = tf.constant([0.0, 0.5, -0.0, -0.5], dtype=tf.float32)
    name = "zero_values"
    input_dict = {"x": x.numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.round"] = tf_math_round_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.round' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.round'.")

check_valid('tf.math.round', generated_inputs['tf.math.round'], lib="tf", suffix=0)
