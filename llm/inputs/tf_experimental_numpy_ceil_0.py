
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_ceil_inputs():
    list_of_inputs = []

    tf.experimental.numpy.experimental_enable_numpy_behavior()

    x1 = tf.constant([1.7, 2.2, 3.3, 4.7])
    input_dict1 = {"x": x1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    x2 = tf.constant([-1.7, -2.2, -3.3, -4.7])
    input_dict2 = {"x": x2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    x3 = tf.constant([1.0, 2.0, 3.0, 4.0])
    input_dict3 = {"x": x3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    x4 = tf.constant([[1.1, 2.2], [3.3, 4.4]])
    input_dict4 = {"x": x4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    x5 = tf.constant([[-1.1, -2.2], [-3.3, -4.4]])
    input_dict5 = {"x": x5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    x6 = tf.constant(np.array([1.5, 2.5, 3.5]))
    input_dict6 = {"x": x6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    x7 = tf.constant(np.array([-1.5, -2.5, -3.5]))
    input_dict7 = {"x": x7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    x8 = tf.constant(np.array([0.0, 0.0, 0.0]))
    input_dict8 = {"x": x8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    x9 = tf.constant([[[1.2, 2.3], [3.4, 4.5]], [[5.6, 6.7], [7.8, 8.9]]])
    input_dict9 = {"x": x9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    x10 = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    input_dict10 = {"x": x10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.ceil"] = tf_experimental_numpy_ceil_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.ceil' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.ceil'.")

check_valid('tf.experimental.numpy.ceil', generated_inputs['tf.experimental.numpy.ceil'], lib="tf", suffix=0)
