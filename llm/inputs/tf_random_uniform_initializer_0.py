
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
    input_dict = {"minval": float(minval), "maxval": float(maxval), "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    minval = -1.0
    maxval = 0.0
    seed = 456
    input_dict = {"minval": float(minval), "maxval": float(maxval), "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    minval = -0.5
    maxval = 0.5
    seed = 789
    input_dict = {"minval": float(minval), "maxval": float(maxval), "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    minval = 0.1
    maxval = 0.9
    seed = 101
    input_dict = {"minval": float(minval), "maxval": float(maxval), "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    minval = -2.0
    maxval = 2.0
    seed = 112
    input_dict = {"minval": float(minval), "maxval": float(maxval), "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    minval = -0.01
    maxval = 0.01
    seed = 131
    input_dict = {"minval": float(minval), "maxval": float(maxval), "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    minval = 5.0
    maxval = 10.0
    seed = 414
    input_dict = {"minval": float(minval), "maxval": float(maxval), "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    minval = -10.0
    maxval = -5.0
    seed = 515
    input_dict = {"minval": float(minval), "maxval": float(maxval), "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    minval = 1.5
    maxval = 2.5
    seed = 616
    input_dict = {"minval": float(minval), "maxval": float(maxval), "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    minval = -2.5
    maxval = 1.5
    seed = 717
    input_dict = {"minval": float(minval), "maxval": float(maxval), "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Larger seed value
    minval = 0.2
    maxval = 0.8
    seed = 2**31 - 1
    input_dict = {"minval": float(minval), "maxval": float(maxval), "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 14: Zero seed
    minval = -0.1
    maxval = 0.1
    seed = 0
    input_dict = {"minval": float(minval), "maxval": float(maxval), "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 15: small values for min and max
    minval = 1e-6
    maxval = 1e-5
    seed = 818
    input_dict = {"minval": float(minval), "maxval": float(maxval), "seed": seed}
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
