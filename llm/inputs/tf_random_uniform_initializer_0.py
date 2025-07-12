
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_random_uniform_initializer_inputs():
    list_of_inputs = []

    # Input 1
    minval = 0.0
    maxval = 1.0
    seed = 123
    input_dict = {"minval": float(minval), "maxval": float(maxval), "seed": int(seed)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    minval = -1.0
    maxval = 0.0
    seed = 456
    input_dict = {"minval": float(minval), "maxval": float(maxval), "seed": int(seed)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.random_uniform_initializer"] = tf_random_uniform_initializer_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.random_uniform_initializer' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random_uniform_initializer'.")

check_valid('tf.random_uniform_initializer', generated_inputs['tf.random_uniform_initializer'], lib="tf", suffix=0)
