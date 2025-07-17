
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_scatter_min_inputs():
    list_of_inputs = []

    # Input 1: Basic example with scalar indices
    ref = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([-1.0, 0.5], dtype=np.float32)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Vector indices
    ref = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([[0.5, 1.0], [4.0, 3.0]], dtype=np.float32)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": True, "name": "scatter_min_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Higher rank indices
    ref = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    indices = np.array([[0, 0], [1, 1]], dtype=np.int32)
    updates = np.array([[0.5, 1.0], [6.0, 5.0]], dtype=np.float32)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative updates
    ref = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.int32)
    updates = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": True, "name": "scatter_min_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Integer ref
    ref = np.array([1, 2, 3, 4], dtype=np.int32)
    indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([-1, 0], dtype=np.int32)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different indices
    ref = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    indices = np.array([4, 3, 2, 1, 0], dtype=np.int32)
    updates = np.array([0, -1, -2, -3, -4], dtype=np.int32)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": True, "name": "scatter_min_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty updates
    ref = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    indices = np.array([], dtype=np.int32)
    updates = np.array([], dtype=np.float32)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Zero values
    ref = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.int32)
    updates = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": True, "name": "scatter_min_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Updates with smaller values than ref
    ref = np.array([10, 20, 30], dtype=np.int32)
    indices = np.array([0, 1, 2], dtype=np.int32)
    updates = np.array([5, 15, 25], dtype=np.int32)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: bfloat16 ref
    ref = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16).astype(np.float16)
    indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([-1.0, 0.5], dtype=np.float16).astype(np.float16)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": True, "name": "scatter_min_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: int64 ref
    ref = np.array([1, 2, 3, 4], dtype=np.int64)
    indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([-1, 0], dtype=np.int64)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: int64 indices
    ref = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    indices = np.array([0, 2], dtype=np.int64)
    updates = np.array([-1.0, 0.5], dtype=np.float32)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": True, "name": "scatter_min_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ScatterMin"] = tf_raw_ops_scatter_min_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ScatterMin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ScatterMin'.")

check_valid('tf.raw_ops.ScatterMin', generated_inputs['tf.raw_ops.ScatterMin'], lib="tf", suffix=0)
