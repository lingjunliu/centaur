
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_reduce_any_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([True, False, True])
    axis = 0
    keepdims = False
    name = "reduce_any_1"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[True, False], [False, False]])
    axis = 0
    keepdims = True
    name = "reduce_any_2"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[True, False], [False, True]])
    axis = 1
    keepdims = False
    name = "reduce_any_3"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[[True, False], [False, True]], [[False, True], [True, True]]])
    axis = 0
    keepdims = True
    name = "reduce_any_4"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[[True, False], [False, True]], [[False, True], [True, True]]])
    axis = 1
    keepdims = False
    name = "reduce_any_5"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([[[True, False], [False, True]], [[False, True], [True, True]]])
    axis = 2
    keepdims = True
    name = "reduce_any_6"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([True, True, True])
    axis = 0
    keepdims = False
    name = "reduce_any_7"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([False, False, False])
    axis = 0
    keepdims = True
    name = "reduce_any_8"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([[False, False], [False, False]])
    axis = 0
    keepdims = False
    name = "reduce_any_9"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([[[False, False], [False, False]], [[False, False], [False, False]]])
    axis = 1
    keepdims = True
    name = "reduce_any_10"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_tensor = np.array([[[True, False, True], [False, True, False]], [[False, True, True], [True, True, False]]])
    axis = 2
    keepdims = False
    name = "reduce_any_11"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12
    input_tensor = np.array([[True, False], [False, True]])
    axis = -1
    keepdims = False
    name = "reduce_any_12"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.reduce_any"] = tf_math_reduce_any_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.reduce_any' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.reduce_any'.")

check_valid('tf.math.reduce_any', generated_inputs['tf.math.reduce_any'], lib="tf", suffix=0)
