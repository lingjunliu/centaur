
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_hypot_inputs():
    list_of_inputs = []

    # Input 1: Basic positive values
    x1 = tf.constant(np.array([3.0]), dtype=tf.float32)
    x2 = tf.constant(np.array([4.0]), dtype=tf.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Mixed positive and negative values
    x1 = tf.constant(np.array([-3.0]), dtype=tf.float32)
    x2 = tf.constant(np.array([4.0]), dtype=tf.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Zero values
    x1 = tf.constant(np.array([0.0]), dtype=tf.float32)
    x2 = tf.constant(np.array([0.0]), dtype=tf.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array
    x1 = tf.constant(np.array([3.0, 5.0, 7.0]), dtype=tf.float32)
    x2 = tf.constant(np.array([4.0, 12.0, 24.0]), dtype=tf.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array
    x1 = tf.constant(np.array([[3.0, 5.0], [7.0, 9.0]]), dtype=tf.float32)
    x2 = tf.constant(np.array([[4.0, 12.0], [24.0, 40.0]]), dtype=tf.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different shapes but broadcastable
    x1 = tf.constant(np.array([3.0, 5.0]), dtype=tf.float32)
    x2 = tf.constant(np.array(4.0), dtype=tf.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger values
    x1 = tf.constant(np.array([300.0]), dtype=tf.float32)
    x2 = tf.constant(np.array([400.0]), dtype=tf.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8: Negative values that become positive after square
    x1 = tf.constant(np.array([-3.0, -5.0]), dtype=tf.float32)
    x2 = tf.constant(np.array([-4.0, -12.0]), dtype=tf.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Different dtypes (float64)
    x1 = tf.constant(np.array([3.0]), dtype=tf.float64)
    x2 = tf.constant(np.array([4.0]), dtype=tf.float64)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D array
    x1 = tf.constant(np.array([[[3.0, 5.0], [7.0, 9.0]],[[1.0, 2.0],[5.0, 4.0]]]), dtype=tf.float32)
    x2 = tf.constant(np.array([[[4.0, 12.0], [24.0, 40.0]],[[2.0, 3.0],[6.0, 7.0]]]), dtype=tf.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.hypot"] = tf_experimental_numpy_hypot_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.hypot' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.hypot'.")

check_valid('tf.experimental.numpy.hypot', generated_inputs['tf.experimental.numpy.hypot'], lib="tf", suffix=0)
