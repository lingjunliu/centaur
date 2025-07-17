
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_scatter_sub_inputs():
    list_of_inputs = []

    # Input 1: Simple case with int32 ref, int32 indices, int32 updates
    ref = tf.Variable(np.array([1, 2, 3, 4, 5], dtype=np.int32))
    indices = np.array([0, 2, 4], dtype=np.int32)
    updates = np.array([5, 3, 1], dtype=np.int32)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": False, "name": "scatter_sub_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32 ref, int64 indices, float32 updates, use_locking=True
    ref = tf.Variable(np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32))
    indices = np.array([1, 3], dtype=np.int64)
    updates = np.array([0.5, 1.5], dtype=np.float32)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": True, "name": "scatter_sub_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int64 ref, int32 indices, int64 updates, negative values
    ref = tf.Variable(np.array([1, 2, 3, 4, 5], dtype=np.int64))
    indices = np.array([0, 1, 4], dtype=np.int32)
    updates = np.array([-1, -2, -3], dtype=np.int64)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": False, "name": "scatter_sub_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multi-dimensional ref, indices, and updates (int32)
    ref = tf.Variable(np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32))
    indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([[5, 5], [1, 1]], dtype=np.int32)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": False, "name": "scatter_sub_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multi-dimensional ref, indices, and updates (float64)
    ref = tf.Variable(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64))
    indices = np.array([1], dtype=np.int32)
    updates = np.array([[0.5, 1.5]], dtype=np.float64)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": True, "name": "scatter_sub_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: uint8
    ref = tf.Variable(np.array([1, 2, 3], dtype=np.uint8))
    indices = np.array([0, 1], dtype=np.int32)
    updates = np.array([1, 1], dtype=np.uint8)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": True, "name": "scatter_sub_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: uint32, int64 indices
    ref = tf.Variable(np.array([1, 2, 3], dtype=np.uint32))
    indices = np.array([0, 1], dtype=np.int64)
    updates = np.array([1, 1], dtype=np.uint32)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": False, "name": "scatter_sub_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8:  indices.shape = [], updates.shape = []
    ref = tf.Variable(np.array([1, 2, 3], dtype=np.int32))
    indices = np.array(1, dtype=np.int32)
    updates = np.array(5, dtype=np.int32)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": False, "name": "scatter_sub_11"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Two dimensional indices
    ref = tf.Variable(np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32))
    indices = np.array([[0], [2]], dtype=np.int32)
    updates = np.array([[5, 5], [1, 1]], dtype=np.int32)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": False, "name": "scatter_sub_12"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: int16
    ref = tf.Variable(np.array([1, 2, 3], dtype=np.int16))
    indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([1, 1], dtype=np.int16)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": False, "name": "scatter_sub_13"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Scalar indices, scalar updates
    ref = tf.Variable(np.array([1, 2, 3], dtype=np.int32))
    indices = np.array(1, dtype=np.int32)
    updates = np.array(10, dtype=np.int32)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": False, "name": "scatter_sub_14"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: Empty indices and updates
    ref = tf.Variable(np.array([1, 2, 3], dtype=np.int32))
    indices = np.array([], dtype=np.int32)
    updates = np.array([], dtype=np.int32)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": False, "name": "scatter_sub_15"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13: Empty updates with proper shape
    ref = tf.Variable(np.array([1, 2, 3], dtype=np.int32))
    indices = np.array([0], dtype=np.int32)
    updates = np.array([], dtype=np.int32)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": False, "name": "scatter_sub_16"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ScatterSub"] = tf_raw_ops_scatter_sub_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ScatterSub' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ScatterSub'.")

check_valid('tf.raw_ops.ScatterSub', generated_inputs['tf.raw_ops.ScatterSub'], lib="tf", suffix=0)
