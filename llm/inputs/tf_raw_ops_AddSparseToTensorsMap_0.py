
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_AddSparseToTensorsMap_inputs():
    list_of_inputs = []

    # Input 1
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

    # Input 2
    sparse_indices = np.array([[0, 1], [1, 0], [2, 2]], dtype=np.int64)
    sparse_values = np.array([5, 6, 7], dtype=np.int32)
    sparse_shape = np.array([3, 3], dtype=np.int64)
    container = ""
    shared_name = ""
    name = "sparse_tensor_map"

    input_dict = {
        "sparse_indices": sparse_indices,
        "sparse_values": sparse_values,
        "sparse_shape": sparse_shape,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    sparse_indices = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 2]], dtype=np.int64)
    sparse_values = np.array([10.0, 11.0, 12.0], dtype=np.float64)
    sparse_shape = np.array([2, 2, 3], dtype=np.int64)
    container = ""
    shared_name = "another_shared_name"
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

    # Input 4
    sparse_indices = np.array([[0], [1], [2]], dtype=np.int64)
    sparse_values = np.array([13, 14, 15], dtype=np.int16)
    sparse_shape = np.array([4], dtype=np.int64)
    container = ""
    shared_name = ""
    name = "op_4"

    input_dict = {
        "sparse_indices": sparse_indices,
        "sparse_values": sparse_values,
        "sparse_shape": sparse_shape,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    sparse_indices = np.array([[0, 0, 0, 0]], dtype=np.int64)
    sparse_values = np.array([16], dtype=np.uint8)
    sparse_shape = np.array([1, 1, 1, 1], dtype=np.int64)
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

    # Input 6
    sparse_indices = np.array([[1, 1], [3, 4], [5, 6]], dtype=np.int64)
    sparse_values = np.array([-1, -2, -3], dtype=np.int32)
    sparse_shape = np.array([7, 7], dtype=np.int64)
    container = ""
    shared_name = "negative_shared_name"
    name = "negative_op"

    input_dict = {
        "sparse_indices": sparse_indices,
        "sparse_values": sparse_values,
        "sparse_shape": sparse_shape,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7
    sparse_indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    sparse_values = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
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

    # Input 8
    sparse_indices = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.int64)
    sparse_values = np.array([1, 2, 3], dtype=np.int64)
    sparse_shape = np.array([3, 3, 3], dtype=np.int64)
    container = ""
    shared_name = "int64_shared_name"
    name = "int64_op"

    input_dict = {
        "sparse_indices": sparse_indices,
        "sparse_values": sparse_values,
        "sparse_shape": sparse_shape,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    sparse_indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    sparse_values = np.array([True, False], dtype=np.bool_)
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

    # Input 10
    sparse_indices = np.array([[0, 0], [1, 2], [2,1]], dtype=np.int64)
    sparse_values = np.array([b"a", b"b", b"c"], dtype=np.object_)
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

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.AddSparseToTensorsMap"] = tf_raw_ops_AddSparseToTensorsMap_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.AddSparseToTensorsMap' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.AddSparseToTensorsMap'.")

check_valid('tf.raw_ops.AddSparseToTensorsMap', generated_inputs['tf.raw_ops.AddSparseToTensorsMap'], lib="tf", suffix=0)
