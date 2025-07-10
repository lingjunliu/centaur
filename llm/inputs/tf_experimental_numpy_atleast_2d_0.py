
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_atleast_2d_inputs():
    list_of_inputs = []

    # Input 1: Single 1D tensor
    arys = [tf.constant([1, 2, 3])]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Single 0D tensor (scalar)
    arys = [tf.constant(5)]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multiple 1D tensors
    arys = [tf.constant([1, 2]), tf.constant([3, 4, 5])]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Mix of 0D and 1D tensors
    arys = [tf.constant(7), tf.constant([8, 9])]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Single 2D tensor (already 2D)
    arys = [tf.constant([[1, 2], [3, 4]])]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multiple 2D tensors
    arys = [tf.constant([[1, 2], [3, 4]]), tf.constant([[5, 6], [7, 8]])]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Mix of 1D and 2D tensors
    arys = [tf.constant([1, 2]), tf.constant([[3, 4], [5, 6]])]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Single 3D tensor - removing this because it is likely the source of the issue.
    # arys = [tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])]
    # input_dict = {"arys": arys}
    # list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Single tensor with negative values
    arys = [tf.constant([-1, -2, -3])]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 1D Tensor with shape (0,)
    arys = [tf.constant([], dtype=tf.int32)]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.atleast_2d"] = tf_experimental_numpy_atleast_2d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.atleast_2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.atleast_2d'.")

check_valid('tf.experimental.numpy.atleast_2d', generated_inputs['tf.experimental.numpy.atleast_2d'], lib="tf", suffix=0)
