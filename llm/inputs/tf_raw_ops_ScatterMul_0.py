
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_scatter_mul_inputs():
    list_of_inputs = []

    # Input 1: Simple case with float32
    ref = tf.Variable(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([2.0, 3.0], dtype=np.float32)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": False, "name": "scatter_mul_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: int32, use_locking=True
    ref = tf.Variable(np.array([1, 2, 3, 4], dtype=np.int32))
    indices = np.array([1, 3], dtype=np.int32)
    updates = np.array([5, 6], dtype=np.int32)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": True, "name": "scatter_mul_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64, multi-dimensional updates
    ref = tf.Variable(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
    indices = np.array([0, 1], dtype=np.int32)
    updates = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float64)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": False, "name": "scatter_mul_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int64, negative indices
    ref = tf.Variable(np.array([1, 2, 3, 4, 5], dtype=np.int64))
    indices = np.array([0, 1, 4], dtype=np.int32)
    updates = np.array([2, 3, 4], dtype=np.int64)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": False, "name": "scatter_mul_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64
    ref = tf.Variable(np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64))
    indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([2+2j, 3+3j], dtype=np.complex64)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": False, "name": "scatter_mul_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: half
    ref = tf.Variable(np.array([1.0, 2.0, 3.0], dtype=np.float16))
    indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([2.0, 3.0], dtype=np.float16)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": False, "name": "scatter_mul_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: uint8
    ref = tf.Variable(np.array([1, 2, 3], dtype=np.uint8))
    indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([2, 3], dtype=np.uint8)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": False, "name": "scatter_mul_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8:  Multi-dimensional ref and updates
    ref = tf.Variable(np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32))
    indices = np.array([0, 1], dtype=np.int32)
    updates = np.array([[[2.0, 3.0], [4.0, 5.0]], [[6.0, 7.0], [8.0, 9.0]]], dtype=np.float32)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": False, "name": "scatter_mul_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: uint32
    ref = tf.Variable(np.array([1, 2, 3], dtype=np.uint32))
    indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([2, 3], dtype=np.uint32)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": False, "name": "scatter_mul_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: uint64
    ref = tf.Variable(np.array([1, 2, 3], dtype=np.uint64))
    indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([2, 3], dtype=np.uint64)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": False, "name": "scatter_mul_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ScatterMul"] = tf_raw_ops_scatter_mul_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ScatterMul' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ScatterMul'.")

check_valid('tf.raw_ops.ScatterMul', generated_inputs['tf.raw_ops.ScatterMul'], lib="tf", suffix=0)
