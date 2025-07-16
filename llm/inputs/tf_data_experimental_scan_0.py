
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_scan_inputs():
    list_of_inputs = []

    # Input 1
    initial_state = np.array(0, dtype=np.int32)
    def scan_func(state, element):
        new_state = tf.add(state, element)
        output_element = new_state
        return new_state, output_element
    input_dict = {"initial_state": initial_state, "scan_func": [scan_func]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    initial_state = np.array([0, 0], dtype=np.int32)
    def scan_func(state, element):
        new_state = tf.add(state, element)
        output_element = new_state * 2
        return new_state, output_element
    input_dict = {"initial_state": initial_state, "scan_func": [scan_func]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    initial_state = np.array([[0, 0], [0, 0]], dtype=np.float32)
    def scan_func(state, element):
        new_state = tf.add(state, element)
        output_element = new_state - element
        return new_state, output_element
    input_dict = {"initial_state": initial_state, "scan_func": [scan_func]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    initial_state = np.array([1, 2, 3], dtype=np.int64)
    def scan_func(state, element):
        new_state = tf.multiply(state, element)
        output_element = new_state + 1
        return new_state, output_element
    input_dict = {"initial_state": initial_state, "scan_func": [scan_func]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    initial_state = np.array(-1, dtype=np.int32)
    def scan_func(state, element):
        new_state = tf.subtract(state, element)
        output_element = new_state * -1
        return new_state, output_element
    input_dict = {"initial_state": initial_state, "scan_func": [scan_func]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    initial_state = np.array([[-1, -2], [-3, -4]], dtype=np.float64)
    def scan_func(state, element):
        new_state = tf.multiply(state, element)
        output_element = new_state + 1
        return new_state, output_element
    input_dict = {"initial_state": initial_state, "scan_func": [scan_func]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    initial_state = np.array([1, 2], dtype=np.int32)
    def scan_func(state, element):
        new_state = tf.add(state, tf.subtract(element[0], element[1]))
        output_element = new_state
        return new_state, output_element
    input_dict = {"initial_state": initial_state, "scan_func": [scan_func]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    initial_state = np.array(1.0, dtype=np.float32)
    def scan_func(state, element):
        new_state = tf.add(state, element)
        output_element = tf.multiply(new_state, element)
        return new_state, output_element
    input_dict = {"initial_state": initial_state, "scan_func": [scan_func]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    initial_state = np.array([0.5, 0.6], dtype=np.float64)
    def scan_func(state, element):
        new_state = tf.add(state, element)
        output_element = tf.multiply(new_state, 2.0)
        return new_state, output_element
    input_dict = {"initial_state": initial_state, "scan_func": [scan_func]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    initial_state = np.array([[-1, 1], [-2, 2]], dtype=np.int32)
    def scan_func(state, element):
        new_state = tf.add(state, element)
        output_element = new_state
        return new_state, output_element
    input_dict = {"initial_state": initial_state, "scan_func": [scan_func]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.scan"] = tf_data_experimental_scan_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.data.experimental.scan' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.scan'.")

check_valid('tf.data.experimental.scan', generated_inputs['tf.data.experimental.scan'], lib="tf", suffix=0)
