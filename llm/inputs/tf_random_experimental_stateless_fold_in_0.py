
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_random_experimental_stateless_fold_in_inputs():
    list_of_inputs = []

    # Input 1
    seed = np.array([1, 2], dtype=np.int32)
    data = np.int32(3)
    alg = 'auto_select'
    input_dict = {"seed": seed, "data": data, "alg": alg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    seed = np.array([100, 200], dtype=np.int32)
    data = np.int32(-5)
    alg = 'philox'
    input_dict = {"seed": seed, "data": data, "alg": alg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    seed = np.array([12345, 67890], dtype=np.int64)
    data = np.int64(1000)
    alg = 'threefry'
    input_dict = {"seed": seed, "data": data, "alg": alg}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.random.experimental.stateless_fold_in"] = tf_random_experimental_stateless_fold_in_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.random.experimental.stateless_fold_in' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.experimental.stateless_fold_in'.")

check_valid('tf.random.experimental.stateless_fold_in', generated_inputs['tf.random.experimental.stateless_fold_in'], lib="tf", suffix=0)
