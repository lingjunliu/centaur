
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_cross_inputs():
    list_of_inputs = []

    # Input 1
    a = tf.constant(np.array([1, 2, 3]))
    b = tf.constant(np.array([4, 5, 6]))
    axisa = -1
    axisb = -1
    axisc = -1
    axis = None

    input_dict = {
        "a": a,
        "b": b,
        "axisa": axisa,
        "axisb": axisb,
        "axisc": axisc,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    a = tf.constant(np.array([[1, 2, 3], [4, 5, 6]]))
    b = tf.constant(np.array([[4, 5, 6], [7, 8, 9]]))
    axisa = -1
    axisb = -1
    axisc = -1
    axis = None

    input_dict = {
        "a": a,
        "b": b,
        "axisa": axisa,
        "axisb": axisb,
        "axisc": axisc,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.cross"] = tf_experimental_numpy_cross_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.cross' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.cross'.")

check_valid('tf.experimental.numpy.cross', generated_inputs['tf.experimental.numpy.cross'], lib="tf", suffix=0)
