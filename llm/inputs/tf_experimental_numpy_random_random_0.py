
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_experimental_numpy_random_random_inputs():
    list_of_inputs = []

    tf.random.set_seed(1)

    # Input 1
    size = tuple()
    input_dict = {"size": size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    size = (1,)
    input_dict = {"size": size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    size = (5,)
    input_dict = {"size": size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    size = (2, 3)
    input_dict = {"size": size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    size = (1, 2, 3)
    input_dict = {"size": size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    size = (3, 2, 1)
    input_dict = {"size": size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    size = (2, 2, 2, 2)
    input_dict = {"size": size}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    size = (10, 10)
    input_dict = {"size": size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    size = (5, 5, 5)
    input_dict = {"size": size}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    size = (4, 3, 2, 1)
    input_dict = {"size": size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.random.random"] = tf_experimental_numpy_random_random_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.random.random' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.random.random'.")

check_valid('tf.experimental.numpy.random.random', generated_inputs['tf.experimental.numpy.random.random'], lib="tf", suffix=0)
