
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_random_uniform_initializer_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {"minval": float(0.0), "maxval": float(1.0), "seed": 123}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {"minval": float(-1.0), "maxval": float(0.0), "seed": 456}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {"minval": float(-0.5), "maxval": float(0.5), "seed": 789}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {"minval": float(0.25), "maxval": float(0.75), "seed": 101}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {"minval": float(-0.75), "maxval": float(-0.25), "seed": 202}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {"minval": float(-2.0), "maxval": float(2.0), "seed": 303}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {"minval": float(1.0), "maxval": float(5.0), "seed": 404}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {"minval": float(-5.0), "maxval": float(-1.0), "seed": 505}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {"minval": float(0.001), "maxval": float(0.002), "seed": 606}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {"minval": float(-0.002), "maxval": float(-0.001), "seed": 707}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_dict = {"minval": float(10.0), "maxval": float(20.0), "seed": 808}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12
    input_dict = {"minval": float(-20.0), "maxval": float(-10.0), "seed": 909}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.random_uniform_initializer"] = tf_random_uniform_initializer_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.random_uniform_initializer' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random_uniform_initializer'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.random_uniform_initializer', generated_inputs['tf.random_uniform_initializer'], lib="tf", suffix=0)
