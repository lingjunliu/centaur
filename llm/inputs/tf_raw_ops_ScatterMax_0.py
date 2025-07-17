
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_scatter_max_inputs():
    list_of_inputs = []

    # Input 1: Basic example with float32
    ref = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([5.0, 6.0], dtype=np.float32)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Int64 with use_locking
    ref = np.array([1, 2, 3, 4], dtype=np.int64)
    indices = np.array([1, 3], dtype=np.int64)
    updates = np.array([5, 6], dtype=np.int64)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": True, "name": "scatter_max_op"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Higher rank ref
    ref = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([[7.0, 8.0], [9.0, 10.0]], dtype=np.float32)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Overlapping indices
    ref = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    indices = np.array([0, 0, 1], dtype=np.int32)
    updates = np.array([5.0, 6.0, 7.0], dtype=np.float32)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5: Negative values
    ref = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    updates = np.array([5.0, -1.0], dtype=np.float32)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty updates
    ref = np.array([1.0, 2.0], dtype=np.float32)
    indices = np.array([], dtype=np.int32)
    updates = np.array([], dtype=np.float32)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float16
    ref = np.array([1.0, 2.0], dtype=np.float16)
    indices = np.array([0, 1], dtype=np.int32)
    updates = np.array([3.0, 4.0], dtype=np.float16)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Updates.shape = []
    ref = np.array([1.0, 2.0], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    updates = np.array(5.0, dtype=np.float32)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D indices and updates
    ref = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    indices = np.array([[0], [2]], dtype=np.int32)
    updates = np.array([[7, 8], [9, 10]], dtype=np.int32)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different data types
    ref = np.array([1, 2, 3, 4], dtype=np.int32)
    indices = np.array([0, 2], dtype=np.int64)
    updates = np.array([5, 6], dtype=np.int32)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ScatterMax"] = tf_raw_ops_scatter_max_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ScatterMax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ScatterMax'.")

check_valid('tf.raw_ops.ScatterMax', generated_inputs['tf.raw_ops.ScatterMax'], lib="tf", suffix=0)
