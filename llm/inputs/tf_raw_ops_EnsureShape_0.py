
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ensure_shape_inputs():
    list_of_inputs = []

    # Input 1: Valid shape, 1D array
    input_tensor = np.array([1, 2, 3], dtype=np.int32)
    shape = [3]
    name = "ensure_shape_1"
    input_dict = {"input": input_tensor, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Valid shape, 2D array
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.float32)
    shape = [2, 2]
    name = "ensure_shape_2"
    input_dict = {"input": input_tensor, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Valid shape, 3D array
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    shape = [2, 2, 2]
    name = "ensure_shape_3"
    input_dict = {"input": input_tensor, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Valid shape, different dtype
    input_tensor = np.array([True, False, True], dtype=np.bool_)
    shape = [3]
    name = "ensure_shape_4"
    input_dict = {"input": input_tensor, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Valid shape, empty array
    input_tensor = np.array([], dtype=np.int32)
    shape = [0]
    name = "ensure_shape_5"
    input_dict = {"input": input_tensor, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Valid shape, scalar (0-dimensional array)
    input_tensor = np.array(5, dtype=np.int32)
    shape = []
    name = "ensure_shape_6"
    input_dict = {"input": input_tensor, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Valid shape, 2D array with different shape
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    shape = [2, 3]
    name = "ensure_shape_7"
    input_dict = {"input": input_tensor, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Valid shape, 1D array with negative values
    input_tensor = np.array([-1, -2, -3], dtype=np.int32)
    shape = [3]
    name = "ensure_shape_8"
    input_dict = {"input": input_tensor, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Valid shape, large array
    input_tensor = np.random.rand(10, 10)
    shape = [10, 10]
    name = "ensure_shape_9"
    input_dict = {"input": input_tensor, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Valid shape, different dtype, complex numbers
    input_tensor = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex128)
    shape = [3]
    name = "ensure_shape_10"
    input_dict = {"input": input_tensor, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.EnsureShape"] = tf_raw_ops_ensure_shape_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.EnsureShape' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.EnsureShape'.")

check_valid('tf.raw_ops.EnsureShape', generated_inputs['tf.raw_ops.EnsureShape'], lib="tf", suffix=0)
