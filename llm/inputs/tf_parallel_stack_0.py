
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_parallel_stack_inputs():
    list_of_inputs = []

    # Input 1: Basic example with integers
    values = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
    name = "stack_int"
    input_dict = {"values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Example with floats
    values = [np.array([1.1, 2.2, 3.3]), np.array([4.4, 5.5, 6.6])]
    name = "stack_float"
    input_dict = {"values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Example with multidimensional arrays
    values = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    name = "stack_multidimensional"
    input_dict = {"values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Example with 3D arrays
    values = [np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])]
    name = "stack_3d"
    input_dict = {"values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Example with negative numbers
    values = [np.array([-1, -2, -3]), np.array([-4, -5, -6])]
    name = "stack_negative"
    input_dict = {"values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Example with zeros
    values = [np.array([0, 0, 0]), np.array([0, 0, 0])]
    name = "stack_zeros"
    input_dict = {"values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Longer list of values
    values = [np.array([i, i+1]) for i in range(3)] #reduced to 3 to avoid timeout
    name = "stack_longer"
    input_dict = {"values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8: Example with string name
    values = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    name = "parallel_stack_example"
    input_dict = {"values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Example with shape (2, 1)
    values = [np.array([[1], [2]]), np.array([[3], [4]])]
    name = "stack_2_1"
    input_dict = {"values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

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
