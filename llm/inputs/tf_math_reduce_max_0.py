
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_reduce_max_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D array
    input_tensor = np.array([1, 5, 2, 8, 3], dtype=np.int32)
    axis = [0]
    keepdims = False
    name = "max_reduction_1"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, reduce along axis 0
    input_tensor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    axis = [0]
    keepdims = False
    name = "max_reduction_2"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, reduce along axis 1, keepdims=True
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    axis = [1]
    keepdims = True
    name = "max_reduction_3"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, reduce along multiple axes
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    axis = [0, 1]
    keepdims = False
    name = "max_reduction_4"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative values and float type
    input_tensor = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    axis = [0]
    keepdims = False
    name = "max_reduction_5"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: All dimensions reduced
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.int32)
    axis = [0, 1]
    keepdims = False
    name = "max_reduction_6"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Keepdims true
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.int32)
    axis = [0]
    keepdims = True
    name = "max_reduction_7"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8: Empty tensor
    input_tensor = np.array([], dtype=np.int32)
    axis = [0]
    keepdims = False
    name = "max_reduction_8"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D tensor, different axes
    input_tensor = np.random.rand(2, 3, 4).astype(np.float32)
    axis = [1, 2]
    keepdims = True
    name = "max_reduction_9"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Tensor with large values
    input_tensor = np.array([[1000, 2000], [3000, 4000]], dtype=np.int32)
    axis = [1]
    keepdims = False
    name = "max_reduction_10"
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
