
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_reduce_max_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D array, no axis specified
    input_tensor = np.array([1, 5, 2, 8, 3], dtype=np.int32)
    axis = None
    keepdims = False
    name = "reduce_max_1"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, axis=0
    input_tensor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    axis = [0]
    keepdims = False
    name = "reduce_max_2"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, axis=1
    input_tensor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    axis = [1]
    keepdims = False
    name = "reduce_max_3"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, axis=[0, 1]
    input_tensor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    axis = [0, 1]
    keepdims = False
    name = "reduce_max_4"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array, axis=0, keepdims=True
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    axis = [0]
    keepdims = True
    name = "reduce_max_5"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array, axis= (0, 2)
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    axis = [0, 2]
    keepdims = False
    name = "reduce_max_6"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array, axis = (1,2), keepdims = True
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    axis = [1, 2]
    keepdims = True
    name = "reduce_max_7"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative values, 2D array, axis=0
    input_tensor = np.array([[-1, -2, -3], [-4, -5, -6]], dtype=np.int32)
    axis = [0]
    keepdims = False
    name = "reduce_max_8"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Float values with nan and inf
    input_tensor = np.array([[1.0, np.nan, 3.0], [np.inf, 5.0, 6.0]], dtype=np.float32)
    axis = [0]
    keepdims = False
    name = "reduce_max_9"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D array, negative axis
    input_tensor = np.array([1, 5, 2, 8, 3], dtype=np.int32)
    axis = [-1]
    keepdims = False
    name = "reduce_max_10"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.reduce_max"] = tf_math_reduce_max_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.reduce_max' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.reduce_max'.")

check_valid('tf.math.reduce_max', generated_inputs['tf.math.reduce_max'], lib="tf", suffix=0)
