
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
    def scan_func1(state, element):
        new_state = state + element
        output_element = element * 2
        return new_state, output_element
    
    input_dict = {
        "initial_state": initial_state,
        "scan_func": [scan_func1]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    initial_state = np.array([0, 0], dtype=np.float32)
    def scan_func2(state, element):
        new_state = state + element
        output_element = element - state
        return new_state, output_element

    input_dict = {
        "initial_state": initial_state,
        "scan_func": [scan_func2]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    initial_state = np.array([[0, 0], [0, 0]], dtype=np.int64)
    def scan_func3(state, element):
        new_state = state + element
        output_element = element * state
        return new_state, output_element

    input_dict = {
        "initial_state": initial_state,
        "scan_func": [scan_func3]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    initial_state = np.array(-1, dtype=np.int32)
    def scan_func4(state, element):
        new_state = state * element
        output_element = state + element
        return new_state, output_element

    input_dict = {
        "initial_state": initial_state,
        "scan_func": [scan_func4]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    initial_state = np.array([1, 2, 3], dtype=np.float64)
    def scan_func5(state, element):
        new_state = state - element
        output_element = state / (element + 1e-8)
        return new_state, output_element
        
    input_dict = {
        "initial_state": initial_state,
        "scan_func": [scan_func5]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    initial_state = np.array([[-1, -2], [-3, -4]], dtype=np.int32)
    def scan_func6(state, element):
        new_state = state + element
        output_element = state * element
        return new_state, output_element

    input_dict = {
        "initial_state": initial_state,
        "scan_func": [scan_func6]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    initial_state = np.array(1.5, dtype=np.float32)
    def scan_func7(state, element):
        new_state = state * element
        output_element = state + element
        return new_state, output_element

    input_dict = {
        "initial_state": initial_state,
        "scan_func": [scan_func7]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    initial_state = np.array([1, 0, -1], dtype=np.int32)
    def scan_func8(state, element):
        new_state = state + element
        output_element = state * element
        return new_state, output_element

    input_dict = {
        "initial_state": initial_state,
        "scan_func": [scan_func8]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    initial_state = np.array([0, 0], dtype=np.float32)
    def scan_func9(state, element):
        new_state = state + element*2
        output_element = state + element
        return new_state, output_element

    input_dict = {
        "initial_state": initial_state,
        "scan_func": [scan_func9]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    initial_state = np.array([[1, 2], [3, 4]], dtype=np.int32)
    def scan_func10(state, element):
        new_state = state + element
        output_element = state * element
        return new_state, output_element
    
    input_dict = {
        "initial_state": initial_state,
        "scan_func": [scan_func10]
    }
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
