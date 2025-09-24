
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_scalar_mul_inputs():
    list_of_inputs = []

    # Input 1: Basic positive scalar and tensor
    scalar = tf.constant(2.0, dtype=tf.float32)
    x = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    name = "basic_positive"
    input_dict = {"scalar": scalar, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative scalar and tensor
    scalar = tf.constant(-3.0, dtype=tf.float32)
    x = tf.constant([[1.0, -2.0], [-3.0, 4.0]], dtype=tf.float32)
    name = "negative_values"
    input_dict = {"scalar": scalar, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Scalar with integer type, tensor with integer type
    scalar = tf.constant(5, dtype=tf.int32)
    x = tf.constant([[1, 2], [3, 4]], dtype=tf.int32)
    name = "integer_type"
    input_dict = {"scalar": scalar, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Scalar with complex type, tensor with complex type
    scalar = tf.constant(1 + 1j, dtype=tf.complex64)
    x = tf.constant([[1 + 2j, 2 - 1j], [3 + 0j, 4 - 3j]], dtype=tf.complex64)
    name = "complex_type"
    input_dict = {"scalar": scalar, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Scalar as float64 and tensor as float64
    scalar = tf.constant(2.5, dtype=tf.float64)
    x = tf.constant([[1.5, 2.5], [3.5, 4.5]], dtype=tf.float64)
    name = "float64_type"
    input_dict = {"scalar": scalar, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D Tensor
    scalar = tf.constant(0.5, dtype=tf.float32)
    x = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=tf.float32)
    name = "3d_tensor"
    input_dict = {"scalar": scalar, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty tensor
    scalar = tf.constant(2.0, dtype=tf.float32)
    x = tf.constant([], dtype=tf.float32, shape=(0, 2))
    name = "empty_tensor"
    input_dict = {"scalar": scalar, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Scalar 0
    scalar = tf.constant(0.0, dtype=tf.float32)
    x = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    name = "scalar_zero"
    input_dict = {"scalar": scalar, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D Tensor
    scalar = tf.constant(3.0, dtype=tf.float32)
    x = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
    name = "1d_tensor"
    input_dict = {"scalar": scalar, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Large values
    scalar = tf.constant(1e9, dtype=tf.float32)
    x = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    name = "large_values"
    input_dict = {"scalar": scalar, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.scalar_mul"] = tf_math_scalar_mul_inputs()
for i in range(len(generated_inputs["tf.math.scalar_mul"])):
    generated_inputs["tf.math.scalar_mul"][i]["x"] = np.array(generated_inputs["tf.math.scalar_mul"][i]["x"])
    generated_inputs["tf.math.scalar_mul"][i]["scalar"] = np.array(generated_inputs["tf.math.scalar_mul"][i]["scalar"])

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.scalar_mul' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.scalar_mul'.")

check_valid('tf.math.scalar_mul', generated_inputs['tf.math.scalar_mul'], lib="tf", suffix=0)
