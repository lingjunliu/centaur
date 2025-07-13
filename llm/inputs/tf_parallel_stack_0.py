
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_parallel_stack_inputs():
    list_of_inputs = []

    def create_input_dict(values, name):
        return {"values": values, "name": name}

    # Input 1: Basic example with integers
    values = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    list_of_inputs.append(copy.deepcopy(create_input_dict(values, "stack_example_1")))

    # Input 2: Example with floats
    values = [np.array([1.0, 2.5, 3.0]), np.array([4.0, 5.5, 6.0])]
    list_of_inputs.append(copy.deepcopy(create_input_dict(values, "stack_example_2")))

    # Input 3: Example with 2D arrays (matrices)
    values = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    list_of_inputs.append(copy.deepcopy(create_input_dict(values, "stack_example_3")))

    # Input 4: Example with negative numbers
    values = [np.array([-1, -2, -3]), np.array([-4, -5, -6])]
    list_of_inputs.append(copy.deepcopy(create_input_dict(values, "stack_example_6")))

    # Input 5: Example with zeros
    values = [np.array([0, 0, 0]), np.array([0, 0, 0])]
    list_of_inputs.append(copy.deepcopy(create_input_dict(values, "stack_example_7")))

    # Input 6: Different data type (int32)
    values = [np.array([1, 2, 3], dtype=np.int32), np.array([4, 5, 6], dtype=np.int32)]
    list_of_inputs.append(copy.deepcopy(create_input_dict(values, "stack_example_12")))

    # Input 7: 2D arrays with float data type
    values = [np.array([[1.5, 2.5], [3.5, 4.5]]), np.array([[5.5, 6.5], [7.5, 8.5]])]
    list_of_inputs.append(copy.deepcopy(create_input_dict(values, "stack_example_14")))

    # Input 8: bool array
    values = [np.array([True, False]), np.array([False, True])]
    list_of_inputs.append(copy.deepcopy(create_input_dict(values, "stack_example_15")))

    # Input 9: Mix of positive and negative
    values = [np.array([-1, 2, -3]), np.array([4, -5, 6])]
    list_of_inputs.append(copy.deepcopy(create_input_dict(values, "stack_example_11")))
    
    # Input 10: Empty list of arrays
    values = []
    list_of_inputs.append(copy.deepcopy(create_input_dict(values, "stack_example_18")))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.parallel_stack"] = tf_parallel_stack_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.parallel_stack' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.parallel_stack'.")

check_valid('tf.parallel_stack', generated_inputs['tf.parallel_stack'], lib="tf", suffix=0)
