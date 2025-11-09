
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_IndexedSlices_inputs():
    list_of_inputs = []

    # Input 1
    values = np.array([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]], dtype=np.float32)
    indices = np.array([1, 3], dtype=np.int32)
    dense_shape = (5, 3)
    input_dict = {"values": values, "indices": indices, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    values = np.array([[[1.0, -2.0],
                        [3.5, 4.5]],
                       [[-1.2, 0.0],
                        [2.2, -3.4]],
                       [[5.5, 6.6],
                        [-7.7, 8.8]]], dtype=np.float64)
    indices = np.array([0, 2, 4], dtype=np.int64)
    dense_shape = (6, 2, 2)
    input_dict = {"values": values, "indices": indices, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    values = np.array([10, -20, 30, -40], dtype=np.int32)
    indices = np.array([2, 0, 5, 3], dtype=np.int32)
    dense_shape = (6,)
    input_dict = {"values": values, "indices": indices, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    values = np.array([[[[1, 2],
                         [3, 4]],
                        [[5, 6],
                         [7, 8]]]], dtype=np.int64)
    indices = np.array([7], dtype=np.int32)
    dense_shape = (8, 2, 2, 2)
    input_dict = {"values": values, "indices": indices, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    values = np.array([[1+2j, -3+4j, 5-6j, 7+0j, -1-1j],
                       [0+0j, 2+2j, -2-3j, 4+5j, -5+4j],
                       [9-1j, -8+2j, 7-3j, -6+6j, 5-5j]], dtype=np.complex64)
    indices = np.array([1, 1, 4], dtype=np.int32)
    dense_shape = (6, 5)
    input_dict = {"values": values, "indices": indices, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    values = np.array([[[True], [False], [True]],
                       [[False], [True], [False]]], dtype=bool)
    indices = np.array([0, 3], dtype=np.int64)
    dense_shape = (4, 3, 1)
    input_dict = {"values": values, "indices": indices, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 (empty slices)
    values = np.empty((0, 4), dtype=np.float16)
    indices = np.empty((0,), dtype=np.int32)
    dense_shape = (10, 4)
    input_dict = {"values": values, "indices": indices, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 (uint8 with duplicates)
    values = np.array([[255],
                       [0],
                       [128]], dtype=np.uint8)
    indices = np.array([9, 0, 9], dtype=np.int32)
    dense_shape = (10, 1)
    input_dict = {"values": values, "indices": indices, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    values = np.array([[[ -1.0,  2.0, -3.0],
                        [  4.5, -5.5,  6.5]]], dtype=np.float32)
    indices = np.array([5], dtype=np.int64)
    dense_shape = (7, 2, 3)
    input_dict = {"values": values, "indices": indices, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    values = np.array([1+2j, -3+4j], dtype=np.complex128)
    indices = np.array([2, 4], dtype=np.int32)
    dense_shape = (5,)
    input_dict = {"values": values, "indices": indices, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11 (zero-sized inner dimension)
    values = np.empty((2, 0), dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    dense_shape = (3, 0)
    input_dict = {"values": values, "indices": indices, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12 (5D, int8)
    values = np.array([[[[[ -1 ]]]],
                       [[[[  2 ]]]]], dtype=np.int8)
    indices = np.array([3, 0], dtype=np.int64)
    dense_shape = (4, 1, 1, 1, 1)
    input_dict = {"values": values, "indices": indices, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.IndexedSlices_3"] = tf_IndexedSlices_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.IndexedSlices_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.IndexedSlices_3'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.IndexedSlices', generated_inputs['tf.IndexedSlices_3'], lib="tf", suffix=3)
