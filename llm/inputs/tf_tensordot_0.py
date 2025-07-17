
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_tensordot_inputs():
    list_of_inputs = []

    # Input 1: Basic matrix multiplication
    a = np.array([[1, 2], [3, 4]], dtype=np.int32)
    b = np.array([[5, 6], [7, 8]], dtype=np.int32)
    axes = 1
    name = "matrix_mult"
    input_dict = {"a": a, "b": b, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Outer product
    a = np.array([1, 2, 3], dtype=np.int32)
    b = np.array([4, 5], dtype=np.int32)
    axes = 0
    name = "outer_product"
    input_dict = {"a": a, "b": b, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensors
    a = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
    b = np.arange(12, dtype=np.int32).reshape(3, 2, 2)
    axes = 1
    name = "3d_tensors"
    input_dict = {"a": a, "b": b, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different data type (float64)
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    axes = 1
    name = "float64_tensors"
    input_dict = {"a": a, "b": b, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Higher axes value - Adjusted for compatibility
    a = np.arange(12, dtype=np.int32).reshape(3, 4)
    b = np.arange(20, dtype=np.int32).reshape(4, 5)
    axes = 1
    name = "higher_axes"
    input_dict = {"a": a, "b": b, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Simple vector dot product
    a = np.array([1, 2, 3], dtype=np.int32)
    b = np.array([4, 5, 6], dtype=np.int32)
    axes = 1
    name = "vector_dot"
    input_dict = {"a": a, "b": b, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger tensors
    a = np.random.rand(5, 6, 7).astype(np.float32)
    b = np.random.rand(7, 8, 9).astype(np.float32)
    axes = 1
    name = "large_tensors"
    input_dict = {"a": a, "b": b, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: axes = 0 with different dimensions
    a = np.array([[1, 2], [3, 4]], dtype=np.int32)
    b = np.array([5, 6], dtype=np.int32)
    axes = 0
    name = "axes_0_diff_dim"
    input_dict = {"a": a, "b": b, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: More complex contraction - Adjusted for compatibility
    a = np.arange(60).reshape(3, 4, 5).astype(np.int32)
    b = np.arange(24).reshape(4, 3, 2).astype(np.int32)
    axes = 2 #changed from 1 to 2
    name = "complex_contraction"
    input_dict = {"a": a, "b": b, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Single element tensors
    a = np.array(5, dtype=np.int32)
    b = np.array(10, dtype=np.int32)
    axes = 0
    name = "single_element"
    input_dict = {"a": a, "b": b, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Adjusted input for matrix multiplication
    a = np.arange(12).reshape(3, 4).astype(np.int32)
    b = np.arange(20).reshape(4, 5).astype(np.int32)
    axes = 1
    name = "adjusted_matrix_mult"
    input_dict = {"a": a, "b": b, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.tensordot"] = tf_tensordot_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.tensordot' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.tensordot'.")

check_valid('tf.tensordot', generated_inputs['tf.tensordot'], lib="tf", suffix=0)
