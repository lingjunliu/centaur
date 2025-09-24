
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_size_inputs():
    list_of_inputs = []

    # Input 1: Scalar tensor, axis=None
    x = tf.constant(5).numpy()
    axis = None
    input_dict = {"x": x, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D tensor, axis=None
    x = tf.constant([1, 2, 3, 4, 5]).numpy()
    axis = None
    input_dict = {"x": x, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D tensor, axis=None
    x = tf.constant([[1, 2], [3, 4]]).numpy()
    axis = None
    input_dict = {"x": x, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D tensor, axis=None
    x = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).numpy()
    axis = None
    input_dict = {"x": x, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tensor with negative values, axis=None
    x = tf.constant([[-1, 2], [-3, 4]]).numpy()
    axis = None
    input_dict = {"x": x, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.size"] = tf_experimental_numpy_size_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.size' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.size'.")

check_valid('tf.experimental.numpy.size', generated_inputs['tf.experimental.numpy.size'], lib="tf", suffix=0)
