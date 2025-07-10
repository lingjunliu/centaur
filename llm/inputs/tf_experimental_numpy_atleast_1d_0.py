
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_experimental_numpy_atleast_1d_inputs():
    list_of_inputs = []

    # Input 1: Single 0-D tensor
    arys = [tf.constant(5)]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Single 1-D tensor
    arys = [tf.constant([1, 2, 3])]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Single 2-D tensor
    arys = [tf.constant([[1, 2], [3, 4]])]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multiple 0-D tensors
    arys = [tf.constant(5), tf.constant(10)]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multiple tensors with different dimensions
    arys = [tf.constant(5), tf.constant([1, 2, 3])]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Single 0-D float tensor
    arys = [tf.constant(5.0)]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Single 0-D bool tensor
    arys = [tf.constant(True)]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Single 1-D string tensor
    arys = [tf.constant(["a", "b", "c"])]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Single 3D tensor
    arys = [tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Tensor with negative values
    arys = [tf.constant([-1, -2, -3])]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.atleast_1d"] = tf_experimental_numpy_atleast_1d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.atleast_1d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.atleast_1d'.")

check_valid('tf.experimental.numpy.atleast_1d', generated_inputs['tf.experimental.numpy.atleast_1d'], lib="tf", suffix=0)
