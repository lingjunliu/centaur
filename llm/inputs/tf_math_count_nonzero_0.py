
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_count_nonzero_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[0, 1, 0], [1, 1, 0]])
    axis = None
    keepdims = False
    dtype = tf.int64
    name = "count_nonzero_1"
    input_dict = {"input": input_tensor, "axis": axis, "keepdims": keepdims, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[0, 1, 0], [1, 1, 0]])
    axis = [0]
    keepdims = False
    dtype = tf.int64
    name = "count_nonzero_2"
    input_dict = {"input": input_tensor, "axis": axis, "keepdims": keepdims, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[0, 1, 0], [1, 1, 0]])
    axis = [1]
    keepdims = True
    dtype = tf.int64
    name = "count_nonzero_3"
    input_dict = {"input": input_tensor, "axis": axis, "keepdims": keepdims, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[0, 1, 0], [1, 1, 0]])
    axis = [0, 1]
    keepdims = False
    dtype = tf.int64
    name = "count_nonzero_4"
    input_dict = {"input": input_tensor, "axis": axis, "keepdims": keepdims, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[-1, 2, -3], [4, -5, 6]])
    axis = None
    keepdims = True
    dtype = tf.int64
    name = "count_nonzero_5"
    input_dict = {"input": input_tensor, "axis": axis, "keepdims": keepdims, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([[[0, 1], [2, 0]], [[3, 0], [0, 4]]])
    axis = [0, 1]
    keepdims = False
    dtype = tf.int64
    name = "count_nonzero_6"
    input_dict = {"input": input_tensor, "axis": axis, "keepdims": keepdims, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([[[0, 1], [2, 0]], [[3, 0], [0, 4]]])
    axis = [0, 2]
    keepdims = True
    dtype = tf.int64
    name = "count_nonzero_7"
    input_dict = {"input": input_tensor, "axis": axis, "keepdims": keepdims, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 (string tensor)
    input_tensor = np.array(["", "a", "  ", "b", ""])
    axis = None
    keepdims = False
    dtype = tf.int64
    name = "count_nonzero_8"
    input_dict = {"input": input_tensor, "axis": axis, "keepdims": keepdims, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 (bool tensor)
    input_tensor = np.array([[False, True, False], [True, True, False]])
    axis = [1]
    keepdims = False
    dtype = tf.int64
    name = "count_nonzero_9"
    input_dict = {"input": input_tensor, "axis": axis, "keepdims": keepdims, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([1.0, 0.0, -2.5, 3.2, 0.0])
    axis = []
    keepdims = False
    dtype = tf.int64
    name = "count_nonzero_10"
    input_dict = {"input": input_tensor, "axis": axis, "keepdims": keepdims, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    input_tensor = np.array([1,2,3])
    axis = [0]
    keepdims = True
    dtype = tf.int64
    name = "count_nonzero_11"
    input_dict = {"input": input_tensor, "axis": axis, "keepdims": keepdims, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12
    input_tensor = np.array([[1, 2, 3], [4, 0, 6]])
    axis = [0]
    keepdims = False
    dtype = tf.int64
    name = "count_nonzero_12"
    input_dict = {"input": input_tensor, "axis": axis, "keepdims": keepdims, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 13
    input_tensor = np.array([[[1, 0], [0, 1]], [[1, 1], [0, 0]]])
    axis = [1, 2]
    keepdims = False
    dtype = tf.int64
    name = "count_nonzero_13"
    input_dict = {"input": input_tensor, "axis": axis, "keepdims": keepdims, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.count_nonzero"] = tf_math_count_nonzero_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.count_nonzero' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.count_nonzero'.")

check_valid('tf.math.count_nonzero', generated_inputs['tf.math.count_nonzero'], lib="tf", suffix=0)
