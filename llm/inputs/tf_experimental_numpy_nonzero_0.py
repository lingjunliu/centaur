
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_nonzero_inputs():
    list_of_inputs = []

    # Input 1: 1D array with some zeros
    a = tf.constant(np.array([1, 0, 2, 0, 3]))
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array with some zeros
    a = tf.constant(np.array([[1, 0, 2], [0, 3, 4]]))
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array with some zeros
    a = tf.constant(np.array([[[1, 0], [2, 3]], [[0, 4], [5, 6]]]))
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array with all zeros
    a = tf.constant(np.array([0, 0, 0]))
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array with all zeros
    a = tf.constant(np.array([[0, 0], [0, 0]]))
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D array with no zeros
    a = tf.constant(np.array([1, 2, 3]))
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array with no zeros
    a = tf.constant(np.array([[1, 2], [3, 4]]))
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Array with negative numbers and zeros
    a = tf.constant(np.array([-1, 0, 2, -3, 0]))
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Boolean array
    a = tf.constant(np.array([True, False, True, True, False]))
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Float array
    a = tf.constant(np.array([1.0, 0.0, 2.5, 0.0, -1.0]))
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.nonzero"] = tf_experimental_numpy_nonzero_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.nonzero' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.nonzero'.")

check_valid('tf.experimental.numpy.nonzero', generated_inputs['tf.experimental.numpy.nonzero'], lib="tf", suffix=0)
