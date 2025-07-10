
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_dynamic_stitch_inputs():
    list_of_inputs = []

    # Input 1: Simple case with scalar indices
    indices = [np.array([0], dtype=np.int32), np.array([1], dtype=np.int32)]
    data = [np.array([10], dtype=np.float32), np.array([20], dtype=np.float32)]
    input_dict = {"indices": indices, "data": data, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Vector indices
    indices = [np.array([0, 2], dtype=np.int32), np.array([1, 3], dtype=np.int32)]
    data = [np.array([1, 3], dtype=np.float32), np.array([2, 4], dtype=np.float32)]
    input_dict = {"indices": indices, "data": data, "name": "vector_indices"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multi-dimensional data
    indices = [np.array([[0, 1], [2, 3]], dtype=np.int32)]
    data = [np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)]
    input_dict = {"indices": indices, "data": data, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different shapes for indices, but consistent data shape
    indices = [np.array([0], dtype=np.int32), np.array([1, 2], dtype=np.int32)]
    data = [np.array([[10, 11]], dtype=np.float32), np.array([[20, 21], [30, 31]], dtype=np.float32)]
    input_dict = {"indices": indices, "data": data, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Overlapping indices (later data overwrites earlier)
    indices = [np.array([0, 1], dtype=np.int32), np.array([1, 2], dtype=np.int32)]
    data = [np.array([10, 11], dtype=np.float32), np.array([20, 21], dtype=np.float32)]
    input_dict = {"indices": indices, "data": data, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6: String data type - Corrected string dtype
    indices = [np.array([0, 1], dtype=np.int32), np.array([2], dtype=np.int32)]
    data = [np.array(["a", "b"], dtype=np.dtype('U')), np.array(["c"], dtype=np.dtype('U'))]
    input_dict = {"indices": indices, "data": data, "name": "string_data"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Bool data type
    indices = [np.array([0, 1], dtype=np.int32), np.array([2], dtype=np.int32)]
    data = [np.array([True, False]), np.array([True])]
    input_dict = {"indices": indices, "data": data, "name": "bool_data"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8:  More complex multi-dimensional example
    indices = [np.array([[0, 1], [2, 3]], dtype=np.int32), np.array([[4, 5], [6, 7]], dtype=np.int32)]
    data = [np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32),
            np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.int32)]
    input_dict = {"indices": indices, "data": data, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10:  Name provided as ""
    indices = [np.array([0, 1], dtype=np.int32), np.array([2], dtype=np.int32)]
    data = [np.array([1, 2], dtype=np.int32), np.array([3], dtype=np.int32)]
    input_dict = {"indices": indices, "data": data, "name": ""}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: More complicated shapes
    indices = [np.array([[0, 1], [2, 3]], dtype=np.int32), np.array([[4, 5]], dtype=np.int32)]
    data = [np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32),
            np.array([[[9, 10], [11, 12]]], dtype=np.float32)]
    input_dict = {"indices": indices, "data": data, "name": "complex_shape"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13: Larger indices, different data type
    indices = [np.array([0, 5], dtype=np.int32), np.array([2, 10], dtype=np.int32)]
    data = [np.array([1.5, 2.5], dtype=np.float32), np.array([3.5, 4.5], dtype=np.float32)]
    input_dict = {"indices": indices, "data": data, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 14: empty strings in data
    indices = [np.array([0, 1], dtype=np.int32), np.array([2], dtype=np.int32)]
    data = [np.array(["", ""], dtype=np.dtype('U')), np.array([""], dtype=np.dtype('U'))]
    input_dict = {"indices": indices, "data": data, "name": "empty_string_data"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.dynamic_stitch"] = tf_dynamic_stitch_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.dynamic_stitch' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.dynamic_stitch'.")

check_valid('tf.dynamic_stitch', generated_inputs['tf.dynamic_stitch'], lib="tf", suffix=0)
