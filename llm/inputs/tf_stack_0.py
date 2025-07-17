
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_stack_inputs():
    list_of_inputs = []

    # Input 1: Basic example with axis=0
    x = np.array([1, 2])
    y = np.array([3, 4])
    z = np.array([5, 6])
    values = [x, y, z]
    axis = 0
    name = "stack_example_1"
    input_dict = {'values': values, 'axis': axis, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Axis=1
    x = np.array([1, 2])
    y = np.array([3, 4])
    z = np.array([5, 6])
    values = [x, y, z]
    axis = 1
    name = "stack_example_2"
    input_dict = {'values': values, 'axis': axis, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D arrays
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[5, 6], [7, 8]])
    values = [x, y]
    axis = 0
    name = "stack_example_3"
    input_dict = {'values': values, 'axis': axis, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Axis=1 with 2D arrays
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[5, 6], [7, 8]])
    values = [x, y]
    axis = 1
    name = "stack_example_4"
    input_dict = {'values': values, 'axis': axis, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D arrays
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    y = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
    values = [x, y]
    axis = 0
    name = "stack_example_6"
    input_dict = {'values': values, 'axis': axis, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Axis = 1 with 3D arrays
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    y = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
    values = [x, y]
    axis = 1
    name = "stack_example_7"
    input_dict = {'values': values, 'axis': axis, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Negative axis
    x = np.array([1, 2])
    y = np.array([3, 4])
    values = [x, y]
    axis = -1
    name = "stack_example_8"
    input_dict = {'values': values, 'axis': axis, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: single dimension arrays, negative axis
    x = np.array([1])
    y = np.array([2])
    z = np.array([3])
    values = [x, y, z]
    axis = -1
    name = "stack_example_11"
    input_dict = {'values': values, 'axis': axis, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: different dtypes, but same dtype and shape
    x = np.array([1, 2], dtype=np.int32)
    y = np.array([3, 4], dtype=np.int32)
    values = [x, y]
    axis = 0
    name = "stack_example_13"
    input_dict = {'values': values, 'axis': axis, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: empty array
    x = np.array([])
    y = np.array([])
    values = [x,y]
    axis = 0
    name = "stack_example_10"
    input_dict = {'values': values, 'axis': axis, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.stack"] = tf_stack_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.stack' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.stack'.")

check_valid('tf.stack', generated_inputs['tf.stack'], lib="tf", suffix=0)
