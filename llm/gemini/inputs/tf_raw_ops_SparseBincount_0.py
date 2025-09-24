
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_sparse_bincount_inputs():
    list_of_inputs = []

    # Input 1: Basic case with int32 weights
    indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    values = np.array([0, 1, 0, 2], dtype=np.int32)
    dense_shape = np.array([2, 2], dtype=np.int64)
    size = np.array(4, dtype=np.int32)
    weights = np.array([1, 2, 3, 4], dtype=np.int32)
    binary_output = False
    name = "sparse_bincount_1"

    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape,
        "size": size,
        "weights": weights,
        "binary_output": binary_output,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2:  int64 values and weights
    indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    values = np.array([0, 1, 0, 2], dtype=np.int64)
    dense_shape = np.array([2, 2], dtype=np.int64)
    size = np.array(4, dtype=np.int64)
    weights = np.array([1, 2, 3, 4], dtype=np.int64)
    binary_output = True
    name = "sparse_bincount_2"

    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape,
        "size": size,
        "weights": weights,
        "binary_output": binary_output,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32 weights
    indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    values = np.array([0, 1, 0, 2], dtype=np.int32)
    dense_shape = np.array([2, 2], dtype=np.int64)
    size = np.array(4, dtype=np.int32)
    weights = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    binary_output = False
    name = "sparse_bincount_3"

    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape,
        "size": size,
        "weights": weights,
        "binary_output": binary_output,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64 weights
    indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    values = np.array([0, 1, 0, 2], dtype=np.int32)
    dense_shape = np.array([2, 2], dtype=np.int64)
    size = np.array(4, dtype=np.int32)
    weights = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    binary_output = True
    name = "sparse_bincount_4"

    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape,
        "size": size,
        "weights": weights,
        "binary_output": binary_output,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Empty weights tensor
    indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    values = np.array([0, 1, 0, 2], dtype=np.int32)
    dense_shape = np.array([2, 2], dtype=np.int64)
    size = np.array(4, dtype=np.int32)
    weights = np.array([], dtype=np.int32)
    binary_output = False
    name = "sparse_bincount_5"

    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape,
        "size": size,
        "weights": weights,
        "binary_output": binary_output,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different dense shape and size
    indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    values = np.array([0, 1, 0, 2], dtype=np.int32)
    dense_shape = np.array([2, 2], dtype=np.int64)
    size = np.array(5, dtype=np.int32)
    weights = np.array([1, 2, 3, 4], dtype=np.int32)
    binary_output = True
    name = "sparse_bincount_6"

    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape,
        "size": size,
        "weights": weights,
        "binary_output": binary_output,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: size = 1
    indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    values = np.array([0, 0, 0, 0], dtype=np.int32)
    dense_shape = np.array([2, 2], dtype=np.int64)
    size = np.array(1, dtype=np.int32)
    weights = np.array([1, 2, 3, 4], dtype=np.int32)
    binary_output = False
    name = "sparse_bincount_7"

    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape,
        "size": size,
        "weights": weights,
        "binary_output": binary_output,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: binary_output = True
    indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    values = np.array([0, 1, 0, 2], dtype=np.int32)
    dense_shape = np.array([2, 2], dtype=np.int64)
    size = np.array(4, dtype=np.int32)
    weights = np.array([1, 2, 3, 4], dtype=np.int32)
    binary_output = True
    name = "sparse_bincount_8"

    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape,
        "size": size,
        "weights": weights,
        "binary_output": binary_output,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Larger dense shape
    indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    values = np.array([0, 1, 0, 2], dtype=np.int32)
    dense_shape = np.array([3, 3], dtype=np.int64)
    size = np.array(4, dtype=np.int32)
    weights = np.array([1, 2, 3, 4], dtype=np.int32)
    binary_output = False
    name = "sparse_bincount_9"

    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape,
        "size": size,
        "weights": weights,
        "binary_output": binary_output,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 10: different index values
    indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    values = np.array([0, 1, 2, 3], dtype=np.int32)
    dense_shape = np.array([2, 2], dtype=np.int64)
    size = np.array(4, dtype=np.int32)
    weights = np.array([1, 2, 3, 4], dtype=np.int32)
    binary_output = False
    name = "sparse_bincount_10"

    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape,
        "size": size,
        "weights": weights,
        "binary_output": binary_output,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SparseBincount"] = tf_raw_ops_sparse_bincount_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SparseBincount' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseBincount'.")

check_valid('tf.raw_ops.SparseBincount', generated_inputs['tf.raw_ops.SparseBincount'], lib="tf", suffix=0)
