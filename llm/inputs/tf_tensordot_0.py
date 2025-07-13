
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
    name = "matrix_multiply"
    input_dict = {"a": a, "b": b, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Outer product
    a = np.array([1, 2, 3], dtype=np.float32)
    b = np.array([4, 5], dtype=np.float32)
    axes = 0
    name = "outer_product"
    input_dict = {"a": a, "b": b, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Higher order tensors, contraction over one axis
    a = np.arange(24).reshape((2, 3, 4)).astype(np.int64)
    b = np.arange(12).reshape((4, 3)).astype(np.int64)
    axes = 1
    name = "higher_order_contraction"
    input_dict = {"a": a, "b": b, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Contraction over scalar values
    a = np.array(5, dtype=np.float64)
    b = np.array(3, dtype=np.float64)
    axes = 0
    name = "scalar_contraction"
    input_dict = {"a": a, "b": b, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different shapes, axes=0
    a = np.array([[1, 2], [3, 4]], dtype=np.int32)
    b = np.array([[5, 6, 7], [8, 9, 10]], dtype=np.int32)
    axes = 0
    name = "diff_shape_outer"
    input_dict = {"a": a, "b": b, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative Values
    a = np.array([[-1, 2], [3, -4]], dtype=np.int32)
    b = np.array([[5, -6], [-7, 8]], dtype=np.int32)
    axes = 1
    name = "negative_values"
    input_dict = {"a": a, "b": b, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: different datatypes (float64)
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    axes = 1
    name = "float64_datatype"
    input_dict = {"a": a, "b": b, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: different datatypes (float64) with different values and axes=0
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    b = np.array([[5.0, -6.0], [-7.0, 8.0]], dtype=np.float64)
    axes = 0
    name = "float64_datatype_outer"
    input_dict = {"a": a, "b": b, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Single element tensors
    a = np.array([[7]], dtype=np.int32)
    b = np.array([[3]], dtype=np.int32)
    axes = 1
    name = "single_element"
    input_dict = {"a": a, "b": b, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: axes = 0 with matching dims
    a = np.arange(12).reshape((2, 2, 3)).astype(np.float32)
    b = np.arange(12).reshape((2, 2, 3)).astype(np.float32)
    axes = 1
    name = "axes_0_match"
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
