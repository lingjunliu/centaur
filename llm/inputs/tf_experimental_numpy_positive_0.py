
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_positive_inputs():
    list_of_inputs = []

    # Input 1: Positive scalar
    x = tf.constant(5)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative scalar
    x = tf.constant(-5)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Zero scalar
    x = tf.constant(0)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D tensor, all positive
    x = tf.constant([1, 2, 3, 4, 5])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D tensor, mixed positive and negative
    x = tf.constant([-1, 2, -3, 4, -5])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.positive"] = tf_experimental_numpy_positive_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.positive' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.positive'.")

check_valid('tf.experimental.numpy.positive', generated_inputs['tf.experimental.numpy.positive'], lib="tf", suffix=0)
