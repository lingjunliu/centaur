
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_reduce_variance_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    axis = 0
    keepdims = False
    name = "variance_0"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    axis = 1
    keepdims = True
    name = "variance_1"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    axis = 0
    keepdims = False
    name = "variance_2"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    axis = 0
    keepdims = True
    name = "variance_3"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    axis = 0
    keepdims = False
    name = "variance_4"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    axis = 1
    keepdims = True
    name = "variance_5"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    axis = 2
    keepdims = False
    name = "variance_6"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    axis = 0
    keepdims = False
    name = "variance_7"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9
    input_tensor = np.array([[[1.0, 2.0,3.0], [4.0, 5.0,6.0]], [[7.0, 8.0,9.0], [10.0, 11.0,12.0]]], dtype=np.float32)
    axis = 2
    keepdims = True
    name = "variance_8"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([[[1.0, 2.0,3.0], [4.0, 5.0,6.0]], [[7.0, 8.0,9.0], [10.0, 11.0,12.0]]], dtype=np.float32)
    axis = 1
    keepdims = False
    name = "variance_9"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    axis = -1
    keepdims = False
    name = "variance_10"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    axis = -2
    keepdims = True
    name = "variance_11"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.reduce_variance"] = tf_math_reduce_variance_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.reduce_variance' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.reduce_variance'.")

check_valid('tf.math.reduce_variance', generated_inputs['tf.math.reduce_variance'], lib="tf", suffix=0)
