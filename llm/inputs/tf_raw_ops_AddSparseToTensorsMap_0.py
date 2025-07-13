
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_AddSparseToTensorsMap_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    sparse_indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    sparse_values = np.array([1.0, 2.0], dtype=np.float32)
    sparse_shape = np.array([3, 4], dtype=np.int64)
    container = ""
    shared_name = ""
    name = None

    input_dict = {
        "sparse_indices": sparse_indices,
        "sparse_values": sparse_values,
        "sparse_shape": sparse_shape,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different shape and values
    sparse_indices = np.array([[0, 1], [2, 0], [2, 3]], dtype=np.int64)
    sparse_values = np.array([5, 6, 7], dtype=np.int32)
    sparse_shape = np.array([3, 4], dtype=np.int64)
    container = "test_container"
    shared_name = "test_shared_name"
    name = "TestName"

    input_dict = {
        "sparse_indices": sparse_indices,
        "sparse_values": sparse_values,
        "sparse_shape": sparse_shape,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Higher rank sparse tensor
    sparse_indices = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.int64)
    sparse_values = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    sparse_shape = np.array([3, 3, 3], dtype=np.int64)
    container = "another_container"
    shared_name = ""
    name = None

    input_dict = {
        "sparse_indices": sparse_indices,
        "sparse_values": sparse_values,
        "sparse_shape": sparse_shape,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Another valid input
    sparse_indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    sparse_values = np.array([1, 2, 3, 4], dtype=np.int64)
    sparse_shape = np.array([2, 2], dtype=np.int64)
    container = ""
    shared_name = ""
    name = None

    input_dict = {
        "sparse_indices": sparse_indices,
        "sparse_values": sparse_values,
        "sparse_shape": sparse_shape,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Sparse tensor with only one element
    sparse_indices = np.array([[0, 0]], dtype=np.int64)
    sparse_values = np.array([10.0], dtype=np.float32)
    sparse_shape = np.array([1, 1], dtype=np.int64)
    container = ""
    shared_name = ""
    name = None

    input_dict = {
        "sparse_indices": sparse_indices,
        "sparse_values": sparse_values,
        "sparse_shape": sparse_shape,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty sparse tensor
    sparse_indices = np.array([], dtype=np.int64).reshape(0, 2)
    sparse_values = np.array([], dtype=np.float32)
    sparse_shape = np.array([2, 3], dtype=np.int64)
    container = ""
    shared_name = ""
    name = None

    input_dict = {
        "sparse_indices": sparse_indices,
        "sparse_values": sparse_values,
        "sparse_shape": sparse_shape,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D sparse tensor with some empty rows/columns
    sparse_indices = np.array([[0, 0, 1], [1, 2, 0], [2, 1, 2]], dtype=np.int64)
    sparse_values = np.array([1, 2, 3], dtype=np.int32)
    sparse_shape = np.array([3, 3, 3], dtype=np.int64)
    container = "test_container2"
    shared_name = "test_shared_name2"
    name = None

    input_dict = {
        "sparse_indices": sparse_indices,
        "sparse_values": sparse_values,
        "sparse_shape": sparse_shape,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different data types
    sparse_indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    sparse_values = np.array([1.5, 2.5], dtype=np.float64)
    sparse_shape = np.array([2, 2], dtype=np.int64)
    container = ""
    shared_name = ""
    name = None

    input_dict = {
        "sparse_indices": sparse_indices,
        "sparse_values": sparse_values,
        "sparse_shape": sparse_shape,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9: 1D Sparse Tensor
    sparse_indices = np.array([[0], [2], [4]], dtype=np.int64)
    sparse_values = np.array([10, 20, 30], dtype=np.int32)
    sparse_shape = np.array([5], dtype=np.int64)
    container = "container1"
    shared_name = "shared_name1"
    name = None
    input_dict = {
        "sparse_indices": sparse_indices,
        "sparse_values": sparse_values,
        "sparse_shape": sparse_shape,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Another 2D example
    sparse_indices = np.array([[0, 1], [1, 0], [2, 2]], dtype=np.int64)
    sparse_values = np.array([-1.0, 2.0, -3.0], dtype=np.float32)
    sparse_shape = np.array([3, 3], dtype=np.int64)
    container = "container2"
    shared_name = "shared_name2"
    name = "op_name"
    input_dict = {
        "sparse_indices": sparse_indices,
        "sparse_values": sparse_values,
        "sparse_shape": sparse_shape,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_raw_ops_AddSparseToTensorsMap_inputs()
generated_inputs["tf.raw_ops.AddSparseToTensorsMap"] = []
for input_dict in inputs:
    generated_inputs["tf.raw_ops.AddSparseToTensorsMap"].append({k: tf.constant(v) if isinstance(v, np.ndarray) else v for k, v in input_dict.items()})

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.AddSparseToTensorsMap' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.AddSparseToTensorsMap'.")

check_valid('tf.raw_ops.AddSparseToTensorsMap', generated_inputs['tf.raw_ops.AddSparseToTensorsMap'], lib="tf", suffix=0)
