
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_polyval_inputs():
    list_of_inputs = []

    # Input 1: Simple polynomial, single point
    p = tf.constant([1, 2, 1], dtype=tf.float32)
    x = tf.constant(1.0, dtype=tf.float32)
    input_dict = {"p": p.numpy(), "x": x.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Simple polynomial, multiple points
    p = tf.constant([1, 2, 1], dtype=tf.float32)
    x = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
    input_dict = {"p": p.numpy(), "x": x.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Higher order polynomial
    p = tf.constant([1, 0, 0, 0, 0, 1], dtype=tf.float32)
    x = tf.constant(2.0, dtype=tf.float32)
    input_dict = {"p": p.numpy(), "x": x.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Polynomial with negative coefficients
    p = tf.constant([1, -2, 1], dtype=tf.float32)
    x = tf.constant(1.0, dtype=tf.float32)
    input_dict = {"p": p.numpy(), "x": x.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Polynomial with integer coefficients
    p = tf.constant([1, 2, 1], dtype=tf.int32)
    x = tf.constant(1, dtype=tf.int32)
    input_dict = {"p": p.numpy(), "x": x.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D input
    p = tf.constant([1, 2, 1], dtype=tf.float32)
    x = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    input_dict = {"p": p.numpy(), "x": x.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Zero Polynomial
    p = tf.constant([0, 0, 0], dtype=tf.float32)
    x = tf.constant(2.0, dtype=tf.float32)
    input_dict = {"p": p.numpy(), "x": x.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Polynomial with x as int
    p = tf.constant([1, 2, 1], dtype=tf.float32)
    x = tf.constant(1, dtype=tf.int32)
    input_dict = {"p": p.numpy(), "x": x.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: p and x are tf.float64
    p = tf.constant([1, 2, 1], dtype=tf.float64)
    x = tf.constant(1.0, dtype=tf.float64)
    input_dict = {"p": p.numpy(), "x": x.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: x has negative values
    p = tf.constant([1, 2, 1], dtype=tf.float32)
    x = tf.constant([-1.0, -2.0, -3.0], dtype=tf.float32)
    input_dict = {"p": p.numpy(), "x": x.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.polyval"] = tf_experimental_numpy_polyval_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.polyval' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.polyval'.")

check_valid('tf.experimental.numpy.polyval', generated_inputs['tf.experimental.numpy.polyval'], lib="tf", suffix=0)
