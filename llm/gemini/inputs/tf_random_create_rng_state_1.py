
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_random_create_rng_state_inputs():
    list_of_inputs = []

    # Input 1: Simple positive seed, philox
    seed = 123
    alg = "philox"
    input_dict = {"seed": seed, "alg": alg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Small negative seed, threefry
    seed = -5
    alg = "threefry"
    input_dict = {"seed": seed, "alg": alg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Zero seed, philox
    seed = 0
    alg = "philox"
    input_dict = {"seed": seed, "alg": alg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Large positive seed, threefry
    seed = 2**31 - 1
    alg = "threefry"
    input_dict = {"seed": seed, "alg": alg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large negative seed, philox
    seed = -(2**31)
    alg = "philox"
    input_dict = {"seed": seed, "alg": alg}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: A different positive seed, threefry
    seed = 987654321
    alg = "threefry"
    input_dict = {"seed": seed, "alg": alg}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Another negative seed, threefry
    seed = -100000
    alg = "threefry"
    input_dict = {"seed": seed, "alg": alg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: small positive seed, philox
    seed = 1
    alg = "philox"
    input_dict = {"seed": seed, "alg": alg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: very large positive seed, philox
    seed = 2147483647
    alg = "philox"
    input_dict = {"seed": seed, "alg": alg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: different alg, threefry
    seed = 55555
    alg = "threefry"
    input_dict = {"seed": seed, "alg": alg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.random.create_rng_state_1"] = tf_random_create_rng_state_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.random.create_rng_state_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.create_rng_state_1'.")

check_valid('tf.random.create_rng_state', generated_inputs['tf.random.create_rng_state_1'], lib="tf", suffix=1)
