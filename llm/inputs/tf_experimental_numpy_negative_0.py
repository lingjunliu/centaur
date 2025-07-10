
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_negative_inputs():
    list_of_inputs = []

    x = tf.constant(np.array([1, 2, 3]), dtype=tf.int32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = tf.constant(np.array([-1, -2, -3]), dtype=tf.int32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = tf.constant(np.array([1.0, 2.0, 3.0]), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = tf.constant(np.array([-1.0, -2.0, -3.0]), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = tf.constant(np.array([[1, 2], [3, 4]]), dtype=tf.int32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = tf.constant(np.array([[-1, -2], [-3, -4]]), dtype=tf.int32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]]), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = tf.constant(np.array([[-1.0, -2.0], [-3.0, -4.0]]), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = tf.constant(np.array([1, 2, -3]), dtype=tf.int32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = tf.constant(np.array([1.5, -2.5, 3.5]), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.negative"] = tf_experimental_numpy_negative_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.negative' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.negative'.")

check_valid('tf.experimental.numpy.negative', generated_inputs['tf.experimental.numpy.negative'], lib="tf", suffix=0)
