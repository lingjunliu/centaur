
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_data_experimental_scan_inputs():
    list_of_inputs = []

    # Input 1
    initial_state = np.array(0, dtype=np.int32)
    def scan_func(state, element):
        new_state = state + element
        output_element = element * 2
        return tf.constant(new_state, dtype=tf.int32), tf.constant(output_element, dtype=tf.int32)
    scan_func_list = [scan_func]

    input_dict = {
        "initial_state": initial_state,
        "scan_func": scan_func_list
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    initial_state = np.array([0, 0], dtype=np.int32)
    def scan_func(state, element):
        new_state = state + element
        output_element = element * 3
        return tf.constant(new_state, dtype=tf.int32), tf.constant(output_element, dtype=tf.int32)
    scan_func_list = [scan_func]

    input_dict = {
        "initial_state": initial_state,
        "scan_func": scan_func_list
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    initial_state = np.array([[0, 0], [0, 0]], dtype=np.int32)
    def scan_func(state, element):
        new_state = state + element
        output_element = element * 4
        return tf.constant(new_state, dtype=tf.int32), tf.constant(output_element, dtype=tf.int32)
    scan_func_list = [scan_func]

    input_dict = {
        "initial_state": initial_state,
        "scan_func": scan_func_list
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    initial_state = np.array(1.0, dtype=np.float32)
    def scan_func(state, element):
        new_state = state + element
        output_element = element * 5
        return tf.constant(new_state, dtype=tf.float32), tf.constant(output_element, dtype=tf.float32)
    scan_func_list = [scan_func]

    input_dict = {
        "initial_state": initial_state,
        "scan_func": scan_func_list
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    initial_state = np.array([1.0, 2.0], dtype=np.float32)
    def scan_func(state, element):
        new_state = state + element
        output_element = element * 6
        return tf.constant(new_state, dtype=tf.float32), tf.constant(output_element, dtype=tf.float32)
    scan_func_list = [scan_func]

    input_dict = {
        "initial_state": initial_state,
        "scan_func": scan_func_list
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    initial_state = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    def scan_func(state, element):
        new_state = state + element
        output_element = element * 7
        return tf.constant(new_state, dtype=tf.float32), tf.constant(output_element, dtype=tf.float32)
    scan_func_list = [scan_func]

    input_dict = {
        "initial_state": initial_state,
        "scan_func": scan_func_list
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    initial_state = np.array(1, dtype=np.int64)
    def scan_func(state, element):
        new_state = state + element
        output_element = element * 8
        return tf.constant(new_state, dtype=tf.int64), tf.constant(output_element, dtype=tf.int64)
    scan_func_list = [scan_func]

    input_dict = {
        "initial_state": initial_state,
        "scan_func": scan_func_list
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    initial_state = np.array([1, 2], dtype=np.int64)
    def scan_func(state, element):
        new_state = state + element
        output_element = element * 9
        return tf.constant(new_state, dtype=tf.int64), tf.constant(output_element, dtype=tf.int64)
    scan_func_list = [scan_func]

    input_dict = {
        "initial_state": initial_state,
        "scan_func": scan_func_list
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    initial_state = np.array([[1, 2], [3, 4]], dtype=np.int64)
    def scan_func(state, element):
        new_state = state + element
        output_element = element * 10
        return tf.constant(new_state, dtype=tf.int64), tf.constant(output_element, dtype=tf.int64)
    scan_func_list = [scan_func]

    input_dict = {
        "initial_state": initial_state,
        "scan_func": scan_func_list
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
   # Input 10
    initial_state = np.array([1, 2, 3], dtype=np.int32)
    def scan_func(state, element):
        new_state = state + element
        output_element = element * 11
        return tf.constant(new_state, dtype=tf.int32), tf.constant(output_element, dtype=tf.int32)
    scan_func_list = [scan_func]

    input_dict = {
        "initial_state": initial_state,
        "scan_func": scan_func_list
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.scan"] = tf_data_experimental_scan_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.scan' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.scan'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.data.experimental.scan', generated_inputs['tf.data.experimental.scan'], lib="tf", suffix=0)
