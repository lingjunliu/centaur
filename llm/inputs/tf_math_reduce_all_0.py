
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_reduce_all_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[True, True], [True, True]])
    axis = 0
    keepdims = False
    name = "reduce_all_1"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[True, False], [True, True]])
    axis = 1
    keepdims = True
    name = "reduce_all_2"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[[True, True], [False, True]], [[True, False], [True, True]]])
    axis = 2
    keepdims = False
    name = "reduce_all_3"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[[True, True], [False, True]], [[True, False], [True, True]]])
    axis = 0
    keepdims = True
    name = "reduce_all_4"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([False, False, False])
    axis = 0
    keepdims = False
    name = "reduce_all_5"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([True, True, True])
    axis = 0
    keepdims = True
    name = "reduce_all_6"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([[[True, True], [True, True]], [[True, True], [True, True]]])
    axis = 1
    keepdims = False
    name = "reduce_all_7"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([[[True, True], [True, True]], [[True, True], [True, True]]])
    axis = -1
    keepdims = True
    name = "reduce_all_8"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([[[True, False], [True, True]], [[True, True], [False, True]]])
    axis = -2
    keepdims = False
    name = "reduce_all_9"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([True, False])
    axis = 0
    keepdims = False
    name = "reduce_all_10"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.reduce_all"] = tf_math_reduce_all_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.reduce_all' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.reduce_all'.")

check_valid('tf.math.reduce_all', generated_inputs['tf.math.reduce_all'], lib="tf", suffix=0)
