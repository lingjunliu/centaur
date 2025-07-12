
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_AddManySparseToTensorsMap_inputs():
    list_of_inputs = []

    # Input 1
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

    # Input 2
    sparse_indices = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1]], dtype=np.int64)
    sparse_values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    sparse_shape = np.array([2, 1, 2], dtype=np.int64)
    container = "my_container"
    shared_name = "my_shared_name"
    name = "my_op_name"

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
    sparse_indices = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]], dtype=np.int64)
    sparse_values = np.array([5, 6, 7, 8, 9, 10], dtype=np.int64)
    sparse_shape = np.array([2, 3], dtype=np.int64)
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

    # Input 4: Empty sparse tensor
    sparse_indices = np.array([], dtype=np.int64).reshape(0, 2)
    sparse_values = np.array([], dtype=np.float64)
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

    # Input 5: single minibatch
    sparse_indices = np.array([[0, 0]], dtype=np.int64)
    sparse_values = np.array([100], dtype=np.int64)
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

    # Input 6: different dtype for values
    sparse_indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    sparse_values = np.array([1.1, 2.2, 3.3, 4.4], dtype=np.float64)
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

    # Input 7: Larger sparse tensor
    sparse_indices = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=np.int64)
    sparse_values = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
    sparse_shape = np.array([2, 2, 2], dtype=np.int64)
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

    # Input 8: Different container and shared name
    sparse_indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    sparse_values = np.array([1, 2, 3, 4], dtype=np.int64)
    sparse_shape = np.array([2, 2], dtype=np.int64)
    container = "test_container"
    shared_name = "test_shared_name"
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

    # Input 9: 3D SparseTensor
    sparse_indices = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [1, 2, 0]], dtype=np.int64)
    sparse_values = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
    sparse_shape = np.array([2, 3, 2], dtype=np.int64)
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

    # Input 10: With name
    sparse_indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    sparse_values = np.array([1, 2, 3, 4], dtype=np.int64)
    sparse_shape = np.array([2, 2], dtype=np.int64)
    container = ""
    shared_name = ""
    name = "test_name"

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
generated_inputs["tf.raw_ops.AddManySparseToTensorsMap"] = tf_raw_ops_AddManySparseToTensorsMap_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.AddManySparseToTensorsMap' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.AddManySparseToTensorsMap'.")

check_valid('tf.raw_ops.AddManySparseToTensorsMap', generated_inputs['tf.raw_ops.AddManySparseToTensorsMap'], lib="tf", suffix=0)
