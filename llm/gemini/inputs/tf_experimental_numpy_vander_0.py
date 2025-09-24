
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_vander_inputs():
    list_of_inputs = []

    # Input 1
    x = tf.constant(np.array([1, 2, 3]))
    N = 3
    increasing = False
    input_dict = {"x": x, "N": N, "increasing": increasing}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = tf.constant(np.array([1, 2, 3]))
    N = 3
    increasing = True
    input_dict = {"x": x, "N": N, "increasing": increasing}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = tf.constant(np.array([1, 2, 3]))
    N = 4
    increasing = False
    input_dict = {"x": x, "N": N, "increasing": increasing}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = tf.constant(np.array([1, 2, 3]))
    N = 4
    increasing = True
    input_dict = {"x": x, "N": N, "increasing": increasing}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = tf.constant(np.array([1, 2, 3]))
    N = 2
    increasing = False
    input_dict = {"x": x, "N": N, "increasing": increasing}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = tf.constant(np.array([1, 2, 3]))
    N = 2
    increasing = True
    input_dict = {"x": x, "N": N, "increasing": increasing}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = tf.constant(np.array([1.5, 2.5, 3.5]))
    N = 3
    increasing = False
    input_dict = {"x": x, "N": N, "increasing": increasing}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = tf.constant(np.array([1.5, 2.5, 3.5]))
    N = 3
    increasing = True
    input_dict = {"x": x, "N": N, "increasing": increasing}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = tf.constant(np.array([-1, -2, -3]))
    N = 3
    increasing = False
    input_dict = {"x": x, "N": N, "increasing": increasing}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = tf.constant(np.array([-1, -2, -3]))
    N = 3
    increasing = True
    input_dict = {"x": x, "N": N, "increasing": increasing}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    x = tf.constant(np.array([1, 2]).astype(np.float32))
    N = 5
    increasing = False
    input_dict = {"x": x, "N": N, "increasing": increasing}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.vander"] = tf_experimental_numpy_vander_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.vander' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.vander'.")

check_valid('tf.experimental.numpy.vander', generated_inputs['tf.experimental.numpy.vander'], lib="tf", suffix=0)
