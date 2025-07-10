
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_random_uniform_initializer_inputs():
    list_of_inputs = []

    # Input 1
    minval = float(-0.1)
    maxval = float(0.1)
    seed = int(1)

    input_dict = {
        "minval": minval,
        "maxval": maxval,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    minval = float(-1.0)
    maxval = float(1.0)
    seed = int(123)

    input_dict = {
        "minval": minval,
        "maxval": maxval,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    minval = float(0.0)
    maxval = float(0.5)
    seed = int(456)

    input_dict = {
        "minval": minval,
        "maxval": maxval,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    minval = float(-5.0)
    maxval = float(2.0)
    seed = int(789)

    input_dict = {
        "minval": minval,
        "maxval": maxval,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    minval = float(-0.001)
    maxval = float(0.001)
    seed = int(101)

    input_dict = {
        "minval": minval,
        "maxval": maxval,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    minval = float(10.0)
    maxval = float(20.0)
    seed = int(112)

    input_dict = {
        "minval": minval,
        "maxval": maxval,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    minval = float(-20.0)
    maxval = float(-10.0)
    seed = int(1233)

    input_dict = {
        "minval": minval,
        "maxval": maxval,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    minval = float(0.0)
    maxval = float(100.0)
    seed = int(4567)

    input_dict = {
        "minval": minval,
        "maxval": maxval,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    minval = float(-100.0)
    maxval = float(0.0)
    seed = int(7890)

    input_dict = {
        "minval": minval,
        "maxval": maxval,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    minval = float(-1000.0)
    maxval = float(1000.0)
    seed = int(1011)

    input_dict = {
        "minval": minval,
        "maxval": maxval,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.random_uniform_initializer"] = tf_random_uniform_initializer_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.random_uniform_initializer' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random_uniform_initializer'.")

check_valid('tf.random_uniform_initializer', generated_inputs['tf.random_uniform_initializer'], lib="tf", suffix=0)
