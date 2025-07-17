
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_random_uniform_initializer_inputs():
    list_of_inputs = []

    # Input 1
    minval = np.float64(-0.05)
    maxval = np.float64(0.05)
    seed = None
    input_dict = {"minval": minval, "maxval": maxval, "seed": seed}
    list_of_inputs.append(input_dict)

    # Input 2
    minval = np.float64(-1.0)
    maxval = np.float64(1.0)
    seed = 123
    input_dict = {"minval": minval, "maxval": maxval, "seed": seed}
    list_of_inputs.append(input_dict)

    # Input 3
    minval = np.float64(0.0)
    maxval = np.float64(10.0)
    seed = 42
    input_dict = {"minval": minval, "maxval": maxval, "seed": seed}
    list_of_inputs.append(input_dict)

    # Input 4
    minval = np.float64(-5.0)
    maxval = np.float64(0.0)
    seed = 99
    input_dict = {"minval": minval, "maxval": maxval, "seed": seed}
    list_of_inputs.append(input_dict)

    # Input 5
    minval = np.float64(0.0)
    maxval = np.float64(0.001)
    seed = 1
    input_dict = {"minval": minval, "maxval": maxval, "seed": seed}
    list_of_inputs.append(input_dict)
    
    # Input 6
    minval = np.float64(-100.0)
    maxval = np.float64(100.0)
    seed = 1000
    input_dict = {"minval": minval, "maxval": maxval, "seed": seed}
    list_of_inputs.append(input_dict)

    # Input 7
    minval = np.float64(-0.5)
    maxval = np.float64(0.5)
    seed = -1
    input_dict = {"minval": minval, "maxval": maxval, "seed": seed}
    list_of_inputs.append(input_dict)

    # Input 8
    minval = np.float64(1.0)
    maxval = np.float64(2.0)
    seed = 2**15 - 1
    input_dict = {"minval": minval, "maxval": maxval, "seed": seed}
    list_of_inputs.append(input_dict)

    # Input 9
    minval = np.float64(-2.0)
    maxval = np.float64(-1.0)
    seed = 0
    input_dict = {"minval": minval, "maxval": maxval, "seed": seed}
    list_of_inputs.append(input_dict)

    # Input 10
    minval = np.float64(10.0)
    maxval = np.float64(10.01)
    seed = 65535
    input_dict = {"minval": minval, "maxval": maxval, "seed": seed}
    list_of_inputs.append(input_dict)

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
