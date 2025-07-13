
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_scatter_nd_update_inputs():
    list_of_inputs = []

    # Input 1
    ref = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
    indices = np.array([[4], [3], [1], [7]], dtype=np.int32)
    updates = np.array([9, 10, 11, 12], dtype=np.int32)
    use_locking = True
    bad_indices_policy = ""
    name = "scatter_nd_update_1"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    ref = np.array([[1, 2], [3, 4]], dtype=np.float32)
    indices = np.array([[0, 0], [1, 1]], dtype=np.int32)
    updates = np.array([5, 6], dtype=np.float32)
    use_locking = False
    bad_indices_policy = ""
    name = "scatter_nd_update_2"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    ref = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    indices = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int64)
    updates = np.array([9, 10], dtype=np.int64)
    use_locking = True
    bad_indices_policy = ""
    name = "scatter_nd_update_3"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    ref = np.array([1, 2, 3, 4], dtype=np.float64)
    indices = np.array([[0], [2]], dtype=np.int64)
    updates = np.array([5, 6], dtype=np.float64)
    use_locking = False
    bad_indices_policy = ""
    name = "scatter_nd_update_4"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    ref = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    indices = np.array([[0, 1], [1, 2]], dtype=np.int32)
    updates = np.array([7, 8], dtype=np.int32)
    use_locking = True
    bad_indices_policy = ""
    name = "scatter_nd_update_5"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    ref = np.array([1, 2, 3, 4], dtype=np.int32)
    indices = np.array([[1], [3]], dtype=np.int32)
    updates = np.array([-5, -6], dtype=np.int32)
    use_locking = False
    bad_indices_policy = ""
    name = "scatter_nd_update_6"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    ref = np.array([[1, 2], [3, 4]], dtype=np.float32)
    indices = np.array([[0, 0], [1, 0]], dtype=np.int32)
    updates = np.array([5.0, 6.0], dtype=np.float32)
    use_locking = True
    bad_indices_policy = ""
    name = "scatter_nd_update_7"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    ref = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    indices = np.array([[0, 0, 0], [1, 0, 1]], dtype=np.int32)
    updates = np.array([9, 10], dtype=np.int32)
    use_locking = False
    bad_indices_policy = ""
    name = "scatter_nd_update_8"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9
    ref = np.array([1, 2, 3], dtype=np.int32)
    indices = np.array([[0], [1], [2]], dtype=np.int32)
    updates = np.array([4, 5, 6], dtype=np.int32)
    use_locking = True
    bad_indices_policy = ""
    name = "scatter_nd_update_9"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    ref = np.array([[1, 2], [3, 4]], dtype=np.int32)
    indices = np.array([[0, 0], [1, 1]], dtype=np.int32)
    updates = np.array([-5, -6], dtype=np.int32)
    use_locking = False
    bad_indices_policy = ""
    name = "scatter_nd_update_10"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ScatterNdUpdate"] = tf_raw_ops_scatter_nd_update_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ScatterNdUpdate' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ScatterNdUpdate'.")

check_valid('tf.raw_ops.ScatterNdUpdate', generated_inputs['tf.raw_ops.ScatterNdUpdate'], lib="tf", suffix=0)
