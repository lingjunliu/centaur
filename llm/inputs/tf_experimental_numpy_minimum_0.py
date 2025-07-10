
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_minimum_inputs():
    list_of_inputs = []

    # Input 1: Basic case with positive integers
    x1 = tf.constant([1, 2, 3, 4, 5])
    x2 = tf.constant([5, 4, 3, 2, 1])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative integers
    x1 = tf.constant([-1, -2, -3, -4, -5])
    x2 = tf.constant([-5, -4, -3, -2, -1])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.minimum"] = tf_experimental_numpy_minimum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.minimum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.minimum'.")

check_valid('tf.experimental.numpy.minimum', generated_inputs['tf.experimental.numpy.minimum'], lib="tf", suffix=0)
