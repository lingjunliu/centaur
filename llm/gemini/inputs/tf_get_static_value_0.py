
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_get_static_value_inputs():
    list_of_inputs = []

    # Input 1: tf.constant
    tensor = tf.constant(10).numpy()
    partial = False
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: tf.constant with partial=True
    tensor = tf.constant(20).numpy()
    partial = True
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: tf.add of two constants
    a = tf.constant(5).numpy()
    b = tf.constant(7).numpy()
    tensor = a + b
    partial = False
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: tf.multiply of two constants with partial=True
    a = tf.constant(3).numpy()
    b = tf.constant(4).numpy()
    tensor = a * b
    partial = True
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: numpy array
    tensor = np.array([1, 2, 3])
    partial = False
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: numpy array and partial=True
    tensor = np.array([4, 5, 6])
    partial = True
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D numpy array
    tensor = np.array([[1, 2], [3, 4]])
    partial = False
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D numpy array and partial=True
    tensor = np.array([[5, 6], [7, 8]])
    partial = True
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: negative value
    tensor = np.array(-5)
    partial = False
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: negative value and partial=True
    tensor = np.array(-10)
    partial = True
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.get_static_value"] = tf_get_static_value_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.get_static_value' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.get_static_value'.")

check_valid('tf.get_static_value', generated_inputs['tf.get_static_value'], lib="tf", suffix=0)
