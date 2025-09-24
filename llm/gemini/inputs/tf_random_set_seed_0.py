
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_random_set_seed_inputs():
    list_of_inputs = []

    input_dict = {"seed": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"seed": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"seed": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"seed": 12345}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"seed": -12345}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"seed": 2**15 - 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"seed": -(2**15)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"seed": np.int32(10)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"seed": np.int64(100)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"seed": np.int32(-10)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.random.set_seed"] = tf_random_set_seed_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.random.set_seed' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.set_seed'.")

check_valid('tf.random.set_seed', generated_inputs['tf.random.set_seed'], lib="tf", suffix=0)
