
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

# Helper scan functions
# These functions are defined to operate on Tensors, as tf.data pipelines
# work with Tensors, not numpy arrays.

def scan_func_sum(state, element):
    new_state = state + element
    return new_state, new_state

def scan_func_prod(state, element):
    new_state = state * element
    return new_state, state

def scan_func_float_sum(state, element):
    new_state = state + element
    return new_state, element

def scan_func_vector_sum(state, element):
    new_state = state + element
    return new_state, state

def scan_func_matrix_sum(state, element):
    new_state = state + element
    return new_state, tf.reduce_sum(state)

def scan_func_nested_tuple(state, element):
    new_sum = state[0] + tf.cast(element, tf.int32)
    new_prod = state[1] * tf.cast(element, tf.float32)
    return (new_sum, new_prod), element

def scan_func_running_avg(state, element):
    element = tf.cast(element, tf.int64)
    new_sum = state['sum'] + element
    new_count = state['count'] + 1
    new_state = {'sum': new_sum, 'count': new_count}
    new_avg = tf.cast(new_sum, tf.float64) / tf.cast(new_count, tf.float64)
    return new_state, new_avg

def scan_func_counter_and_bool(state, element):
    new_state = state + 1
    output_element = tf.equal(element % 2, 0)
    return new_state, output_element

def scan_func_abs_sum(state, element):
    new_state = state + tf.cast(tf.abs(element), state.dtype)
    return new_state, state

def scan_func_3d_tensor(state, element):
    return state + element, tf.reduce_mean(element)

def scan_func_complex_sum(state, element):
    new_state = state + tf.cast(element, state.dtype)
    return new_state, state

def scan_func_counter_and_tuple_output(state, element):
    new_state = state + 1
    output = (element, element * 2, element ** 2)
    return new_state, output


def tf_data_experimental_scan_inputs():
    list_of_inputs = []

    # Input 1: Simple integer summation
    input_dict_1 = {
        'initial_state': np.int32(0),
        'scan_func': [scan_func_sum],
        'inner_values': (np.array([1, 2, 3, 4, 5], dtype=np.int32),)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Running product with int64
    input_dict_2 = {
        'initial_state': np.int64(1),
        'scan_func': [scan_func_prod],
        'inner_values': (np.array([1, 2, 3, 4], dtype=np.int64),)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Float summation
    input_dict_3 = {
        'initial_state': np.float32(0.0),
        'scan_func': [scan_func_float_sum],
        'inner_values': (np.array([0.1, 0.2, 0.3], dtype=np.float32),)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Vector state
    input_dict_4 = {
        'initial_state': np.array([0, 0], dtype=np.int32),
        'scan_func': [scan_func_vector_sum],
        'inner_values': (np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32),)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Matrix state with float64
    input_dict_5 = {
        'initial_state': np.zeros((2, 3), dtype=np.float64),
        'scan_func': [scan_func_matrix_sum],
        'inner_values': (np.ones((4, 2, 3), dtype=np.float64),)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Nested tuple state
    input_dict_6 = {
        'initial_state': (np.int32(0), np.float32(1.0)),
        'scan_func': [scan_func_nested_tuple],
        'inner_values': (np.array([1, 2, 3], dtype=np.int32),)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Nested dictionary state for running average
    input_dict_7 = {
        'initial_state': {'sum': np.int64(0), 'count': np.int64(0)},
        'scan_func': [scan_func_running_avg],
        'inner_values': (np.array([10, 20, 30, 40], dtype=np.int32),)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Counter state, boolean output
    input_dict_8 = {
        'initial_state': np.int32(0),
        'scan_func': [scan_func_counter_and_bool],
        'inner_values': (np.arange(10, dtype=np.int32),)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Negative initial state
    input_dict_9 = {
        'initial_state': np.array([-100], dtype=np.int32),
        'scan_func': [scan_func_abs_sum],
        'inner_values': (np.array([-1, 2, -3, 4], dtype=np.int32),)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: 3D Tensor state
    input_dict_10 = {
        'initial_state': np.zeros((2, 2, 2), dtype=np.float32),
        'scan_func': [scan_func_3d_tensor],
        'inner_values': (np.random.rand(5, 2, 2, 2).astype(np.float32),)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Complex number state
    input_dict_11 = {
        'initial_state': np.complex64(1+0j),
        'scan_func': [scan_func_complex_sum],
        'inner_values': (np.array([1+2j, 3+4j], dtype=np.complex64),)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    # Input 12: State is a counter, output is a tuple
    input_dict_12 = {
        'initial_state': np.uint8(0),
        'scan_func': [scan_func_counter_and_tuple_output],
        'inner_values': (np.array([5, 10, 15], dtype=np.uint8),)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_12))

    return list_of_inputs

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

check_valid('tf.data.experimental.scan', generated_inputs['tf.data.experimental.scan'], lib="tf", suffix=0)
