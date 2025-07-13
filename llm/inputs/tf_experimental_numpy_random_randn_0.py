
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.random.set_seed(1)

def tf_experimental_numpy_random_randn_inputs():
    list_of_inputs = []

    # Input 1: Single integer
    args = (np.int64(5),)
    input_dict = {"args": args}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Two integers
    args = (np.int64(2), np.int64(3))
    input_dict = {"args": args}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Three integers
    args = (np.int64(2), np.int64(3), np.int64(4))
    input_dict = {"args": args}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large integers
    args = (np.int64(100), np.int64(100))
    input_dict = {"args": args}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: One
    args = (np.int64(1),)
    input_dict = {"args": args}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Square matrix
    args = (np.int64(7), np.int64(7))
    input_dict = {"args": args}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D tensor
    args = (np.int64(2), np.int64(3), np.int64(2), np.int64(2))
    input_dict = {"args": args}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 5D tensor
    args = (np.int64(1), np.int64(2), np.int64(3), np.int64(4), np.int64(5))
    input_dict = {"args": args}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different sized dims
    args = (np.int64(10), np.int64(5), np.int64(2))
    input_dict = {"args": args}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.random.randn"] = tf_experimental_numpy_random_randn_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.random.randn' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.random.randn'.")

check_valid('tf.experimental.numpy.random.randn', generated_inputs['tf.experimental.numpy.random.randn'], lib="tf", suffix=0)
