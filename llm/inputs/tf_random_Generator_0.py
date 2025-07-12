
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_random_Generator_inputs():
    list_of_inputs = []

    # Input 1
    state = np.array([1, 0], dtype=np.int64)
    copy_from = None
    alg = "philox"

    input_dict = {
        "copy_from": copy_from,
        "state": state,
        "alg": alg
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    state = np.array([1234, 0], dtype=np.int64)
    copy_from = None
    alg = "philox"

    input_dict = {
        "copy_from": copy_from,
        "state": state,
        "alg": alg
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    state = np.array([5678, 0, 0], dtype=np.int64)
    copy_from = None
    alg = "threefry"

    input_dict = {
        "copy_from": copy_from,
        "state": state,
        "alg": alg
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    state = np.array([9101, 0], dtype=np.int64)
    copy_from = None
    alg = "philox"

    input_dict = {
        "copy_from": copy_from,
        "state": state,
        "alg": alg
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    state = np.array([11213, 0, 0], dtype=np.int64)
    copy_from = None
    alg = "threefry"

    input_dict = {
        "copy_from": copy_from,
        "state": state,
        "alg": alg
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    state = np.array([1415, 0], dtype=np.int64)
    copy_from = None
    alg = "philox"

    input_dict = {
        "copy_from": copy_from,
        "state": state,
        "alg": alg
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    state = np.array([0, 0, 0], dtype=np.int64)
    copy_from = None
    alg = "threefry"

    input_dict = {
        "copy_from": copy_from,
        "state": state,
        "alg": alg
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8
    state = np.array([2**31 - 1, 0], dtype=np.int64)
    copy_from = None
    alg = "philox"

    input_dict = {
        "copy_from": copy_from,
        "state": state,
        "alg": alg
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    state = np.array([-1, 0, 0], dtype=np.int64)
    copy_from = None
    alg = "threefry"

    input_dict = {
        "copy_from": copy_from,
        "state": state,
        "alg": alg
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    state = np.array([1, 2], dtype=np.int64)
    copy_from = None
    alg = "philox"

    input_dict = {
        "copy_from": copy_from,
        "state": state,
        "alg": alg
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

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
