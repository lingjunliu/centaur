
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_random_generator_inputs():
    list_of_inputs = []

    # Input 1: Valid inputs with alg="philox"
    state = np.array([1, 0], dtype=np.int64)
    input_dict = {
        "copy_from": None,
        "state": state,
        "alg": "philox"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Valid inputs with alg="threefry"
    state = np.array([1, 0, 0], dtype=np.int64)
    input_dict = {
        "copy_from": None,
        "state": state,
        "alg": "threefry"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Valid inputs with copy_from and alg
    g = tf.random.Generator.from_seed(1234)
    input_dict = {
        "copy_from": str(g),
        "state": None,
        "alg": "philox"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Valid inputs with copy_from and alg
    g = tf.random.Generator.from_seed(4321, alg='threefry')
    input_dict = {
        "copy_from": str(g),
        "state": None,
        "alg": "threefry"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Valid inputs with numpy state and alg
    state = np.array([1, 2], dtype=np.int64)
    input_dict = {
        "copy_from": None,
        "state": state,
        "alg": "philox"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: None Alg, String copy from
    g = tf.random.Generator.from_seed(1234, alg='philox')
    input_dict = {
        "copy_from": str(g),
        "state": None,
        "alg": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: All non-None, philox
    g = tf.random.Generator.from_seed(42)
    state = np.array([5, 6], dtype=np.int64)

    input_dict = {
        "copy_from": str(g),
        "state": state,
        "alg": "philox"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Just copy_from, no alg, no state
    g = tf.random.Generator.from_seed(123)
    input_dict = {
        "copy_from": str(g),
        "state": None,
        "alg": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: copy_from & state
    g = tf.random.Generator.from_seed(123)
    state = np.array([7, 8], dtype=np.int64)
    input_dict = {
        "copy_from": str(g),
        "state": state,
        "alg": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: No state or Alg, only copy from
    g = tf.random.Generator.from_seed(123, alg="threefry")
    input_dict = {
        "copy_from": str(g),
        "state": None,
        "alg": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.random.Generator"] = tf_random_generator_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.random.Generator' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.Generator'.")

check_valid('tf.random.Generator', generated_inputs['tf.random.Generator'], lib="tf", suffix=0)
