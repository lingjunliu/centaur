
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_gather_inputs():
    list_of_inputs = []

    # Input 1
    params = np.array(['p0', 'p1', 'p2', 'p3', 'p4', 'p5'], dtype=object)
    indices = np.array([2, 0, 2, 5])
    validate_indices = False
    axis = 0
    batch_dims = 0
    name = "gather_example_1"
    input_dict = {"params": params, "indices": indices, "validate_indices": validate_indices, "axis": axis, "batch_dims": batch_dims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    params = np.array([[0, 1.0, 2.0], [10.0, 11.0, 12.0], [20.0, 21.0, 22.0], [30.0, 31.0, 32.0]]).astype(np.float32)
    indices = np.array([3, 1])
    validate_indices = False
    axis = 0
    batch_dims = 0
    name = "gather_example_2"
    input_dict = {"params": params, "indices": indices, "validate_indices": validate_indices, "axis": axis, "batch_dims": batch_dims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    params = np.array([[0, 1.0, 2.0], [10.0, 11.0, 12.0], [20.0, 21.0, 22.0], [30.0, 31.0, 32.0]]).astype(np.float32)
    indices = np.array([2, 1])
    validate_indices = False
    axis = 1
    batch_dims = 0
    name = "gather_example_3"
    input_dict = {"params": params, "indices": indices, "validate_indices": validate_indices, "axis": axis, "batch_dims": batch_dims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    params = np.random.rand(5, 6, 7, 8).astype(np.float32)
    indices = np.random.randint(0, 7, size=(10, 11)).astype(np.int32)
    validate_indices = False
    axis = 2
    batch_dims = 0
    name = "gather_example_4"
    input_dict = {"params": params, "indices": indices, "validate_indices": validate_indices, "axis": axis, "batch_dims": batch_dims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    params = np.array([[0, 0, 1, 0, 2], [3, 0, 0, 0, 4], [0, 5, 0, 6, 0]]).astype(np.int32)
    indices = np.array([[2, 4], [0, 4], [1, 3]])
    validate_indices = False
    axis = 1
    batch_dims = 1
    name = "gather_example_5"
    input_dict = {"params": params, "indices": indices, "validate_indices": validate_indices, "axis": axis, "batch_dims": batch_dims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative axis
    params = np.random.rand(5, 6, 7).astype(np.float32)
    indices = np.array([0, 2, 4])
    validate_indices = False
    axis = -1
    batch_dims = 0
    name = "gather_example_6"
    input_dict = {"params": params, "indices": indices, "validate_indices": validate_indices, "axis": axis, "batch_dims": batch_dims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Multi-dimensional indices
    params = np.random.rand(5, 6).astype(np.float32)
    indices = np.array([[0, 1], [2, 3]])
    validate_indices = False
    axis = 0
    batch_dims = 0
    name = "gather_example_7"
    input_dict = {"params": params, "indices": indices, "validate_indices": validate_indices, "axis": axis, "batch_dims": batch_dims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: batch_dims > 0 with multi-dimensional indices
    params = np.random.rand(2, 3, 4).astype(np.float32)
    indices = np.random.randint(0, 4, size=(2,3,2)).astype(np.int32)
    validate_indices = False
    axis = 2
    batch_dims = 1
    name = "gather_example_8"
    input_dict = {"params": params, "indices": indices, "validate_indices": validate_indices, "axis": axis, "batch_dims": batch_dims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D params
    params = np.array([10, 20, 30, 40, 50]).astype(np.int32)
    indices = np.array([1, 3, 0])
    validate_indices = False
    axis = 0
    batch_dims = 0
    name = "gather_example_9"
    input_dict = {"params": params, "indices": indices, "validate_indices": validate_indices, "axis": axis, "batch_dims": batch_dims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: params with large values, indices with large values within range
    params = np.arange(1000).reshape((10, 100)).astype(np.int32)
    indices = np.array([99, 50, 10, 0])
    validate_indices = False
    axis = 1
    batch_dims = 0
    name = "gather_example_10"
    input_dict = {"params": params, "indices": indices, "validate_indices": validate_indices, "axis": axis, "batch_dims": batch_dims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    params = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.int32)
    indices = np.array([[0, 1], [1, 0]]).astype(np.int32)
    validate_indices = False
    axis = 0
    batch_dims = 0
    name = "gather_example_11"
    input_dict = {"params": params, "indices": indices, "validate_indices": validate_indices, "axis": axis, "batch_dims": batch_dims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    params = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.int32)
    indices = np.array([0, 1]).astype(np.int32)
    validate_indices = False
    axis = 0
    batch_dims = 0
    name = "gather_example_12"
    input_dict = {"params": params, "indices": indices, "validate_indices": validate_indices, "axis": axis, "batch_dims": batch_dims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.gather"] = tf_gather_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.gather' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.gather'.")

check_valid('tf.gather', generated_inputs['tf.gather'], lib="tf", suffix=0)
