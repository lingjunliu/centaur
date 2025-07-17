
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_random_normal_initializer_inputs():
    list_of_inputs = []

    def create_input(mean, stddev, seed):
      return {"mean": mean, "stddev": stddev, "seed": seed}

    # Input 1
    list_of_inputs.append(create_input(float(0.0), float(0.05), None))

    # Input 2
    list_of_inputs.append(create_input(float(1.0), float(0.1), 123))

    # Input 3
    list_of_inputs.append(create_input(float(-1.0), float(0.01), 456))

    # Input 4
    list_of_inputs.append(create_input(float(0.5), float(0.2), 789))

    # Input 5
    list_of_inputs.append(create_input(float(-0.5), float(0.02), 101))

    # Input 6
    list_of_inputs.append(create_input(float(2.0), float(0.5), 202))

    # Input 7
    list_of_inputs.append(create_input(float(-2.0), float(0.005), 303))

    # Input 8
    list_of_inputs.append(create_input(float(0.1), float(1.0), 404))

    # Input 9
    list_of_inputs.append(create_input(float(-0.1), float(2.0), 505))

    # Input 10
    list_of_inputs.append(create_input(float(10.0), float(5.0), 606))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.random_normal_initializer"] = tf_random_normal_initializer_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.random_normal_initializer' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random_normal_initializer'.")

check_valid('tf.random_normal_initializer', generated_inputs['tf.random_normal_initializer'], lib="tf", suffix=0)
