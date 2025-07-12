
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_atleast_3d_inputs():
    list_of_inputs = []

    # Input 1: Scalar
    arys = [np.array(1)]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array
    arys = [np.array([1, 2, 3])]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array
    arys = [np.array([[1, 2], [3, 4]])]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Already 3D array
    arys = [np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.atleast_3d"] = tf_experimental_numpy_atleast_3d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.atleast_3d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.atleast_3d'.")

check_valid('tf.experimental.numpy.atleast_3d', generated_inputs['tf.experimental.numpy.atleast_3d'], lib="tf", suffix=0)
