
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_random_Generator_inputs():
    list_of_inputs = []

    # Input 1: Valid input with seed and philox algorithm
    state1 = np.array([123, 0, 0], dtype=np.int64)
    input_dict1 = {
        "copy_from": None,
        "state": state1,
        "alg": "philox"
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Valid input with seed and threefry algorithm
    state2 = np.array([123, 0], dtype=np.int64)
    input_dict2 = {
        "copy_from": None,
        "state": state2,
        "alg": "threefry"
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Valid input with state containing negative values and philox algorithm
    state3 = np.array([-123, 0, 0], dtype=np.int64)
    input_dict3 = {
        "copy_from": None,
        "state": state3,
        "alg": "philox"
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Valid input with state containing large values and threefry algorithm
    state4 = np.array([2**32, 0], dtype=np.int64)
    input_dict4 = {
        "copy_from": None,
        "state": state4,
        "alg": "threefry"
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5:  Valid input with zero state and philox
    state5 = np.array([0, 0, 0], dtype=np.int64)
    input_dict5 = {
        "copy_from": None,
        "state": state5,
        "alg": "philox"
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 9: Valid input with different state values and threefry algorithm
    state9 = np.array([100, 200], dtype=np.int64)
    input_dict9 = {
        "copy_from": None,
        "state": state9,
        "alg": "threefry"
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Valid input with larger state values and philox
    state10 = np.array([2**62, 2**60, 0], dtype=np.int64)
    input_dict10 = {
        "copy_from": None,
        "state": state10,
        "alg": "philox"
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.random.Generator"] = tf_random_Generator_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.random.Generator' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.Generator'.")

check_valid('tf.random.Generator', generated_inputs['tf.random.Generator'], lib="tf", suffix=0)
