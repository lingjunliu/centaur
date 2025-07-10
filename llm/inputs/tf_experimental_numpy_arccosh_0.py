
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_arccosh_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input, scalar
    x = tf.constant(2.0)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D tensor
    x = tf.constant([2.0, 3.0, 4.0, 5.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D tensor
    x = tf.constant([[2.0, 3.0], [4.0, 5.0]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D tensor
    x = tf.constant([[[2.0, 3.0], [4.0, 5.0]], [[6.0, 7.0], [8.0, 9.0]]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tensor with values close to 1
    x = tf.constant([1.01, 1.1, 1.5, 2.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger values
    x = tf.constant([10.0, 100.0, 1000.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: tf.float64
    x = tf.constant([2.0, 3.0, 4.0], dtype=tf.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: tf.float32
    x = tf.constant([2.0, 3.0, 4.0], dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Tensor with different shape
    x = tf.constant([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Tensor from numpy array
    x = tf.constant(np.array([2.0, 3.0, 4.0]))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Minimum possible value
    x = tf.constant(1.0)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.arccosh"] = tf_experimental_numpy_arccosh_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.arccosh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.arccosh'.")

check_valid('tf.experimental.numpy.arccosh', generated_inputs['tf.experimental.numpy.arccosh'], lib="tf", suffix=0)
