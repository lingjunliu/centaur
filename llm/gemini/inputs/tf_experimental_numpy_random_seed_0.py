
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_experimental_numpy_random_seed_inputs():
    list_of_inputs = []

    # Input 1: Positive integer
    s = 42
    input_dict = {"s": s}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Zero
    s = 0
    input_dict = {"s": s}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative integer
    s = -1
    input_dict = {"s": s}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Large positive integer
    s = 2**31 - 1
    input_dict = {"s": s}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Small positive integer
    s = 1
    input_dict = {"s": s}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Another positive integer
    s = 12345
    input_dict = {"s": s}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Another negative integer
    s = -100
    input_dict = {"s": s}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Different large positive integer
    s = 2147483646
    input_dict = {"s": s}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Small negative integer
    s = -5
    input_dict = {"s": s}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Another Positive integer
    s = 987654321
    input_dict = {"s": s}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.random.seed"] = tf_experimental_numpy_random_seed_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.random.seed' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.random.seed'.")

check_valid('tf.experimental.numpy.random.seed', generated_inputs['tf.experimental.numpy.random.seed'], lib="tf", suffix=0)
