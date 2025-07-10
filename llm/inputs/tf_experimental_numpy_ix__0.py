
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_ix__inputs():
    list_of_inputs = []

    # Input 1, valid
    a = np.array([2, 3, 4])
    b = np.array([7, 8, 9])
    input_dict = {
        "args": [a, b]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid
    a = np.array([1, 5])
    b = np.array([2, 6])
    c = np.array([3, 7])

    input_dict = {
        "args": [a, b, c]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid
    a = np.array([0])
    b = np.array([1])
    c = np.array([2])
    d = np.array([3])

    input_dict = {
        "args": [a, b, c, d]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid
    a = np.array([])
    b = np.array([])

    input_dict = {
        "args": [a, b]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid
    a = np.array([1,2,3])
    input_dict = {
        "args": [a]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid
    a = np.array([0, 1])
    b = np.array([2, 3])
    input_dict = {
        "args": [a, b]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid
    a = np.array([1])
    b = np.array([1])
    c = np.array([1])
    d = np.array([1])
    e = np.array([1])
    f = np.array([1])

    input_dict = {
        "args": [a, b, c, d, e, f]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid
    a = np.array([1,2])
    b = np.array([3,4])
    c = np.array([5,6])
    d = np.array([7,8])
    e = np.array([9,10])
    f = np.array([11,12])
    g = np.array([13,14])
    input_dict = {
        "args": [a, b, c, d, e, f, g]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9, valid
    a = np.array([1])
    b = np.array([2])
    c = np.array([3])
    d = np.array([4])
    e = np.array([5])
    f = np.array([6])
    g = np.array([7])
    h = np.array([8])
    input_dict = {
        "args": [a, b, c, d, e, f, g, h]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid
    a = np.array([1])
    b = np.array([2])
    c = np.array([3])
    d = np.array([4])
    e = np.array([5])
    f = np.array([6])
    g = np.array([7])
    h = np.array([8])
    i = np.array([9])
    input_dict = {
        "args": [a, b, c, d, e, f, g, h, i]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.ix_"] = tf_experimental_numpy_ix__inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.ix_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.ix_'.")

check_valid('tf.experimental.numpy.ix_', generated_inputs['tf.experimental.numpy.ix_'], lib="tf", suffix=0)
