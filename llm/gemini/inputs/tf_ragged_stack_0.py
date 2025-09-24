
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_ragged_stack_inputs():
    list_of_inputs = []

    # Input 1: Basic case with ragged tensors and axis=0
    t1 = tf.constant([[1, 2], [3, 4]])
    t2 = tf.constant([[6,7], [7, 8]])
    values = [t1, t2]
    axis = 0
    name = "stack_example_1"
    input_dict = {"values": values, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic case with ragged tensors and axis=1
    t1 = tf.constant([[1, 2], [3, 4]])
    t2 = tf.constant([[6,7], [7, 8]])
    values = [t1, t2]
    axis = 1
    name = "stack_example_2"
    input_dict = {"values": values, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Dense tensors with different sizes and axis=0
    t3 = tf.constant([[1, 2], [4, 5]])
    t4 = tf.constant([[5,6], [6,7]])
    values = [t3, t4]
    axis = 0
    name = "stack_example_3"
    input_dict = {"values": values, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Stacking dense tensors with same sizes and axis=0
    t5 = tf.constant([[1, 2], [3, 4]])
    t6 = tf.constant([[5, 6], [7, 8]])
    values = [t5, t6]
    axis = 0
    name = "stack_example_4"
    input_dict = {"values": values, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Stacking dense tensors with same sizes and axis=1
    t5 = tf.constant([[1, 2], [3, 4]])
    t6 = tf.constant([[5, 6], [7, 8]])
    values = [t5, t6]
    axis = 1
    name = "stack_example_5"
    input_dict = {"values": values, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Single ragged tensor, axis=0
    t7 = tf.constant([[1, 2], [4,5]])
    values = [t7]
    axis = 0
    name = "stack_example_6"
    input_dict = {"values": values, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Single dense tensor, axis=0
    t8 = tf.constant([[1, 2], [3, 4]])
    values = [t8]
    axis = 0
    name = "stack_example_7"
    input_dict = {"values": values, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: List of scalars, axis=0
    values = [tf.constant(1), tf.constant(2), tf.constant(3)]
    axis = 0
    name = "stack_example_8"
    input_dict = {"values": values, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.ragged.stack"] = tf_ragged_stack_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.ragged.stack' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.ragged.stack'.")

check_valid('tf.ragged.stack', generated_inputs['tf.ragged.stack'], lib="tf", suffix=0)
