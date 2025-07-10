
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_ragged_cross_inputs():
    list_of_inputs = []

    # Input 1
    inputs = [tf.ragged.constant([['a'], ['b', 'c']]),
              tf.ragged.constant([['d'], ['e']]),
              tf.ragged.constant([['f'], ['g']])]
    name = None
    input_dict = {'inputs': inputs, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    inputs = [tf.ragged.constant([['1'], ['2', '3']]),
              tf.ragged.constant([['4'], ['5']])]
    name = "feature_cross"
    input_dict = {'inputs': inputs, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    inputs = [tf.ragged.constant([['a', 'b'], ['c']]),
              tf.ragged.constant([['d'], ['e', 'f']])]
    name = None
    input_dict = {'inputs': inputs, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    inputs = [tf.ragged.constant([['a']]),
              tf.ragged.constant([['b']]),
              tf.ragged.constant([['c']])]
    name = "test_cross"
    input_dict = {'inputs': inputs, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.ragged.cross"] = tf_ragged_cross_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.ragged.cross' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.ragged.cross'.")

check_valid('tf.ragged.cross', generated_inputs['tf.ragged.cross'], lib="tf", suffix=0)
