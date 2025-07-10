
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_special_dawsn_inputs():
    list_of_inputs = []

    # Input 1: Basic test with a scalar
    x = tf.constant(1.0, dtype=tf.float32).numpy()
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Test with a negative scalar
    x = tf.constant(-1.0, dtype=tf.float32).numpy()
    name = "negative_scalar"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Test with a 1D tensor
    x = tf.constant(np.array([-1.0, -0.5, 0.5, 1.0]), dtype=tf.float32).numpy()
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Test with a 2D tensor
    x = tf.constant(np.array([[-1.0, -0.5], [0.5, 1.0]]), dtype=tf.float32).numpy()
    name = "2d_tensor"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Test with a 3D tensor
    x = tf.constant(np.random.rand(2, 3, 4).astype(np.float32)).numpy()
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Test with float64
    x = tf.constant(1.0, dtype=tf.float64).numpy()
    name = "float64_test"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger values
    x = tf.constant(np.array([5.0, 10.0, -5.0, -10.0]), dtype=tf.float32).numpy()
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Values close to zero
    x = tf.constant(np.array([0.001, -0.001, 0.0]), dtype=tf.float32).numpy()
    name = "near_zero"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Mixed positive and negative
    x = tf.constant(np.array([-2.5, 1.5, -0.75, 3.2]), dtype=tf.float32).numpy()
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Tensor with only zeros
    x = tf.constant(np.zeros((2, 2)), dtype=tf.float32).numpy()
    name = "zeros_tensor"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.special.dawsn"] = tf_math_special_dawsn_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.special.dawsn' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.special.dawsn'.")

check_valid('tf.math.special.dawsn', generated_inputs['tf.math.special.dawsn'], lib="tf", suffix=0)
