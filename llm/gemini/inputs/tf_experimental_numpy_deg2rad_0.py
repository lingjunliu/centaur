
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_deg2rad_inputs():
    list_of_inputs = []

    # Input 1: Scalar
    x = tf.constant(0.0)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive scalar
    x = tf.constant(90.0)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative scalar
    x = tf.constant(-180.0)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D Tensor
    x = tf.constant([0.0, 45.0, 90.0, 180.0, 360.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D Tensor
    x = tf.constant([[0.0, 30.0], [60.0, 90.0]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D Tensor
    x = tf.constant([[[0.0, 10.0], [20.0, 30.0]], [[40.0, 50.0], [60.0, 70.0]]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tensor with negative values
    x = tf.constant([-45.0, -90.0, -180.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Tensor with mixed positive and negative values
    x = tf.constant([-45.0, 0.0, 45.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large values
    x = tf.constant([720.0, 1080.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Tensor with float64 dtype
    x = tf.constant(np.array([0.0, 45.0, 90.0]), dtype=tf.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.deg2rad"] = tf_experimental_numpy_deg2rad_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.deg2rad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.deg2rad'.")

check_valid('tf.experimental.numpy.deg2rad', generated_inputs['tf.experimental.numpy.deg2rad'], lib="tf", suffix=0)
