
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nest_flatten_inputs():
    list_of_inputs = []

    # Input 1: Simple list, expand_composites=False
    structure = [1, 2, 3]
    expand_composites = False
    input_dict = {"structure": structure, "expand_composites": expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nest.flatten"] = tf_nest_flatten_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nest.flatten' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nest.flatten'.")

check_valid('tf.nest.flatten', generated_inputs['tf.nest.flatten'], lib="tf", suffix=0)
