
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_sparsebincount_inputs():
    list_of_inputs = []

    # Input 1: Basic case with int32 values and weights
    indices = np.array([[0, 0], [1, 0], [1, 1]]).astype(np.int64)
    values = np.array([0, 1, 2]).astype(np.int32)
    dense_shape = np.array([2, 2]).astype(np.int64)
    size = np.array(4).astype(np.int32)
    weights = np.array([1, 2, 3]).astype(np.int32)
    binary_output = False
    name = "bincount_1"
    input_dict = {"indices": tf.convert_to_tensor(indices, dtype=tf.int64), "values": tf.convert_to_tensor(values, dtype=tf.int32), "dense_shape": tf.convert_to_tensor(dense_shape, dtype=tf.int64), "size": tf.convert_to_tensor(size, dtype=tf.int32), "weights": tf.convert_to_tensor(weights, dtype=tf.int32), "binary_output": binary_output, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic case with int64 values and float32 weights
    indices = np.array([[0, 0], [1, 0], [1, 1]]).astype(np.int64)
    values = np.array([0, 1, 2]).astype(np.int64)
    dense_shape = np.array([2, 2]).astype(np.int64)
    size = np.array(4).astype(np.int64)
    weights = np.array([1.0, 2.0, 3.0]).astype(np.float32)
    binary_output = True
    name = "bincount_2"
    input_dict = {"indices": tf.convert_to_tensor(indices, dtype=tf.int64), "values": tf.convert_to_tensor(values, dtype=tf.int64), "dense_shape": tf.convert_to_tensor(dense_shape, dtype=tf.int64), "size": tf.convert_to_tensor(size, dtype=tf.int64), "weights": tf.convert_to_tensor(weights, dtype=tf.float32), "binary_output": binary_output, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Empty weights (acts as all weights equal to 1)
    indices = np.array([[0, 0], [1, 0], [1, 1]]).astype(np.int64)
    values = np.array([0, 1, 2]).astype(np.int32)
    dense_shape = np.array([2, 2]).astype(np.int64)
    size = np.array(4).astype(np.int32)
    weights = np.array([]).astype(np.int32)
    binary_output = False
    name = "bincount_3"
    input_dict = {"indices": tf.convert_to_tensor(indices, dtype=tf.int64), "values": tf.convert_to_tensor(values, dtype=tf.int32), "dense_shape": tf.convert_to_tensor(dense_shape, dtype=tf.int64), "size": tf.convert_to_tensor(size, dtype=tf.int32), "weights": tf.convert_to_tensor(weights, dtype=tf.int32), "binary_output": binary_output, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Larger size and out-of-range values (ignored)
    indices = np.array([[0, 0], [1, 0], [1, 1]]).astype(np.int64)
    values = np.array([0, 1, 5]).astype(np.int64)
    dense_shape = np.array([2, 2]).astype(np.int64)
    size = np.array(6).astype(np.int64)
    weights = np.array([1.0, 2.0, 3.0]).astype(np.float64)
    binary_output = True
    name = "bincount_4"
    input_dict = {"indices": tf.convert_to_tensor(indices, dtype=tf.int64), "values": tf.convert_to_tensor(values, dtype=tf.int64), "dense_shape": tf.convert_to_tensor(dense_shape, dtype=tf.int64), "size": tf.convert_to_tensor(size, dtype=tf.int64), "weights": tf.convert_to_tensor(weights, dtype=tf.float64), "binary_output": binary_output, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different indices
    indices = np.array([[0, 1], [1, 1], [1, 0]]).astype(np.int64)
    values = np.array([0, 1, 2]).astype(np.int32)
    dense_shape = np.array([2, 2]).astype(np.int64)
    size = np.array(4).astype(np.int32)
    weights = np.array([1, 2, 3]).astype(np.int32)
    binary_output = False
    name = "bincount_5"
    input_dict = {"indices": tf.convert_to_tensor(indices, dtype=tf.int64), "values": tf.convert_to_tensor(values, dtype=tf.int32), "dense_shape": tf.convert_to_tensor(dense_shape, dtype=tf.int64), "size": tf.convert_to_tensor(size, dtype=tf.int32), "weights": tf.convert_to_tensor(weights, dtype=tf.int32), "binary_output": binary_output, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float64 weights
    indices = np.array([[0, 0], [1, 0], [1, 1]]).astype(np.int64)
    values = np.array([0, 1, 2]).astype(np.int64)
    dense_shape = np.array([2, 2]).astype(np.int64)
    size = np.array(4).astype(np.int64)
    weights = np.array([1.5, 2.5, 3.5]).astype(np.float64)
    binary_output = False
    name = "bincount_6"
    input_dict = {"indices": tf.convert_to_tensor(indices, dtype=tf.int64), "values": tf.convert_to_tensor(values, dtype=tf.int64), "dense_shape": tf.convert_to_tensor(dense_shape, dtype=tf.int64), "size": tf.convert_to_tensor(size, dtype=tf.int64), "weights": tf.convert_to_tensor(weights, dtype=tf.float64), "binary_output": binary_output, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: binary_output = True, int64 weights
    indices = np.array([[0, 0], [1, 0], [1, 1]]).astype(np.int64)
    values = np.array([0, 1, 2]).astype(np.int32)
    dense_shape = np.array([2, 2]).astype(np.int64)
    size = np.array(4).astype(np.int32)
    weights = np.array([1, 2, 3]).astype(np.int64)
    binary_output = True
    name = "bincount_7"
    input_dict = {"indices": tf.convert_to_tensor(indices, dtype=tf.int64), "values": tf.convert_to_tensor(values, dtype=tf.int32), "dense_shape": tf.convert_to_tensor(dense_shape, dtype=tf.int64), "size": tf.convert_to_tensor(size, dtype=tf.int32), "weights": tf.convert_to_tensor(weights, dtype=tf.int64), "binary_output": binary_output, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8: dense_shape [1,1], size = 2
    indices = np.array([[0, 0]]).astype(np.int64)
    values = np.array([1]).astype(np.int32)
    dense_shape = np.array([1, 1]).astype(np.int64)
    size = np.array(2).astype(np.int32)
    weights = np.array([2]).astype(np.int32)
    binary_output = False
    name = "bincount_8"
    input_dict = {"indices": tf.convert_to_tensor(indices, dtype=tf.int64), "values": tf.convert_to_tensor(values, dtype=tf.int32), "dense_shape": tf.convert_to_tensor(dense_shape, dtype=tf.int64), "size": tf.convert_to_tensor(size, dtype=tf.int32), "weights": tf.convert_to_tensor(weights, dtype=tf.int32), "binary_output": binary_output, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: int64 values and int64 weights
    indices = np.array([[0, 0], [1, 0], [1, 1]]).astype(np.int64)
    values = np.array([0, 1, 2]).astype(np.int64)
    dense_shape = np.array([2, 2]).astype(np.int64)
    size = np.array(4).astype(np.int64)
    weights = np.array([1, 2, 3]).astype(np.int64)
    binary_output = False
    name = "bincount_9"
    input_dict = {"indices": tf.convert_to_tensor(indices, dtype=tf.int64), "values": tf.convert_to_tensor(values, dtype=tf.int64), "dense_shape": tf.convert_to_tensor(dense_shape, dtype=tf.int64), "size": tf.convert_to_tensor(size, dtype=tf.int64), "weights": tf.convert_to_tensor(weights, dtype=tf.int64), "binary_output": binary_output, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: different values and int32 weights
    indices = np.array([[0, 0], [1, 0], [1, 1]]).astype(np.int64)
    values = np.array([2, 3, 1]).astype(np.int32)
    dense_shape = np.array([2, 2]).astype(np.int64)
    size = np.array(5).astype(np.int32)
    weights = np.array([1, 2, 3]).astype(np.int32)
    binary_output = False
    name = "bincount_10"
    input_dict = {"indices": tf.convert_to_tensor(indices, dtype=tf.int64), "values": tf.convert_to_tensor(values, dtype=tf.int32), "dense_shape": tf.convert_to_tensor(dense_shape, dtype=tf.int64), "size": tf.convert_to_tensor(size, dtype=tf.int32), "weights": tf.convert_to_tensor(weights, dtype=tf.int32), "binary_output": binary_output, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_raw_ops_sparsebincount_inputs()
generated_inputs["tf.raw_ops.SparseBincount"] = inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SparseBincount' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseBincount'.")

check_valid('tf.raw_ops.SparseBincount', generated_inputs['tf.raw_ops.SparseBincount'], lib="tf", suffix=0)
