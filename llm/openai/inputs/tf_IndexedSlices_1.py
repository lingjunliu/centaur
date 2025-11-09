
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import torch
import copy

def tf_IndexedSlices_inputs():
    list_of_inputs = []

    # Input 1
    values = np.array([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]], dtype=np.float32)
    indices = np.array([1, 3], dtype=np.int32)
    dense_shape = np.array([5, 3], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    # Input 2
    values = np.arange(3 * 2 * 2).reshape(3, 2, 2).astype(np.int32)
    indices = np.array([0, 2, 4], dtype=np.int32)
    dense_shape = np.array([6, 2, 2], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    # Input 3
    values = np.array([0.1, -2.3, 4.5, 6.7], dtype=np.float64)
    indices = np.array([9, 1, 7, 3], dtype=np.int64)
    dense_shape = np.array([10], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    # Input 4
    values = np.array([[[True], [False], [True], [True]]], dtype=np.bool_)
    indices = np.array([2], dtype=np.int32)
    dense_shape = np.array([3, 4, 1], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    # Input 5
    values = np.array([[1 + 2j, 3 + 4j],
                       [5 + 6j, 7 + 8j]], dtype=np.complex64)
    indices = np.array([5, 1], dtype=np.int64)
    dense_shape = np.array([10, 2], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    # Input 6 (empty slices)
    values = np.empty((0, 5), dtype=np.float32)
    indices = np.array([], dtype=np.int32)
    dense_shape = np.array([4, 5], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    # Input 7 (duplicate indices)
    values = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]], dtype=np.int64)
    indices = np.array([2, 2, 4], dtype=np.int64)
    dense_shape = np.array([6, 3], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    # Input 8 (float16, 3D values)
    values = np.arange(2 * 4 * 3).reshape(2, 4, 3).astype(np.float16)
    indices = np.array([10, 999], dtype=np.int32)
    dense_shape = np.array([1000, 4, 3], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    # Input 9 (negative indices)
    values = np.array([[1, -1],
                       [127, -128]], dtype=np.int8)
    indices = np.array([-1, -3], dtype=np.int32)
    dense_shape = np.array([5, 2], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    # Input 10 (uint8, 1D values)
    values = np.array([0, 255, 128], dtype=np.uint8)
    indices = np.array([0, 1, 2], dtype=np.int64)
    dense_shape = np.array([3], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    # Input 11 (string dtype)
    values = np.array([["a", "b"], ["c", ""]], dtype=object)
    indices = np.array([1, 4], dtype=np.int32)
    dense_shape = np.array([5, 2], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    # Input 12 (NaN/Inf)
    values = np.array([[[np.nan, np.inf], [-np.inf, 0.0]]], dtype=np.float32)
    indices = np.array([3], dtype=np.int32)
    dense_shape = np.array([8, 2, 2], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    # Input 13 (zero-sized inner dim)
    values = np.empty((4, 0, 3), dtype=np.float32)
    indices = np.array([0, 1, 2, 3], dtype=np.int32)
    dense_shape = np.array([10, 0, 3], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    return list_of_inputs

generated_inputs["tf.IndexedSlices_1"] = tf_IndexedSlices_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.IndexedSlices_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.IndexedSlices_1'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.IndexedSlices', generated_inputs['tf.IndexedSlices_1'], lib="tf", suffix=1)
