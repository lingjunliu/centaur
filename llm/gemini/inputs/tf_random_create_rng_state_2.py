
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_random_create_rng_state_inputs():
    list_of_inputs = []

    # Input 1
    seed = [1234]
    alg = "philox"
    input_dict = {"seed": seed, "alg": alg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    seed = [12, 34]
    alg = "threefry"
    input_dict = {"seed": seed, "alg": alg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    seed = [5]
    alg = "philox"
    input_dict = {"seed": seed, "alg": alg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    seed = [1, 2, 3, 4]
    alg = "philox"
    input_dict = {"seed": seed, "alg": alg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    seed = [-1, -2]
    alg = "threefry"
    input_dict = {"seed": seed, "alg": alg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    seed = [0]
    alg = "philox"
    input_dict = {"seed": seed, "alg": alg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    seed = [1000000000000]
    alg = "philox"
    input_dict = {"seed": seed, "alg": alg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    seed = [1, 2, 3, 4, 5, 6, 7, 8]
    alg = "threefry"
    input_dict = {"seed": seed, "alg": alg}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    seed = [1, -1]
    alg = "philox"
    input_dict = {"seed": seed, "alg": alg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    seed = [-1000]
    alg = "threefry"
    input_dict = {"seed": seed, "alg": alg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.random.create_rng_state_2"] = tf_random_create_rng_state_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.random.create_rng_state_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.create_rng_state_2'.")

check_valid('tf.random.create_rng_state', generated_inputs['tf.random.create_rng_state_2'], lib="tf", suffix=2)
