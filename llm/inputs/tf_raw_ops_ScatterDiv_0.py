
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_scatter_div_inputs():
    list_of_inputs = []

    # Input 1: float32, scalar indices
    ref = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([2.0, 3.0], dtype=np.float32)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": False, "name": "scatter_div_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: int32, vector indices
    ref = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    indices = np.array([[0], [2]], dtype=np.int32)
    updates = np.array([[2, 2], [5, 3]], dtype=np.int32)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": True, "name": "scatter_div_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64, high rank indices
    ref = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64)
    indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    updates = np.array([2.0, 2.0], dtype=np.float64)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": False, "name": "scatter_div_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int64, scalar indices, zero updates
    ref = np.array([1, 2, 3], dtype=np.int64)
    indices = np.array([0, 1], dtype=np.int32)
    updates = np.array([2, 1], dtype=np.int64)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": False, "name": "scatter_div_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32, vector indices
    ref = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    indices = np.array([[0], [1]], dtype=np.int64)
    updates = np.array([[2, 1], [1, 2]], dtype=np.float32)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": True, "name": "scatter_div_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex64
    ref = np.array([1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex64)
    indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([2 + 1j, 1 + 2j], dtype=np.complex64)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": False, "name": "scatter_div_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: uint8
    ref = np.array([1, 2, 3], dtype=np.uint8)
    indices = np.array([0, 1], dtype=np.int64)
    updates = np.array([2, 1], dtype=np.uint8)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": True, "name": "scatter_div_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: int16
    ref = np.array([1, 2, 3], dtype=np.int16)
    indices = np.array([0, 1], dtype=np.int32)
    updates = np.array([2, 1], dtype=np.int16)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": False, "name": "scatter_div_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: int8
    ref = np.array([1, 2, 3], dtype=np.int8)
    indices = np.array([0, 1], dtype=np.int64)
    updates = np.array([2, 1], dtype=np.int8)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": True, "name": "scatter_div_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32, higher rank indices
    ref = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    indices = np.array([[0], [2]], dtype=np.int32)
    updates = np.array([[2.0, 2.0], [1.0, 3.0]], dtype=np.float32)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": False, "name": "scatter_div_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = {}
temp = tf_raw_ops_scatter_div_inputs()

generated_inputs["tf.raw_ops.ScatterDiv"] = []
for i in range(len(temp)):
  input_dict = copy.deepcopy(temp[i])
  input_dict["ref"] = tf.Variable(input_dict["ref"])
  generated_inputs["tf.raw_ops.ScatterDiv"].append(input_dict)

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ScatterDiv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ScatterDiv'.")

check_valid('tf.raw_ops.ScatterDiv', generated_inputs['tf.raw_ops.ScatterDiv'], lib="tf", suffix=0)
