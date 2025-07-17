
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_scatter_nd_sub_inputs():
    list_of_inputs = []

    # Input 1, valid
    ref = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
    indices = np.array([[4], [3], [1], [7]], dtype=np.int32)
    updates = np.array([9, 10, 11, 12], dtype=np.int32)
    use_locking = False
    bad_indices_policy = ""
    name = "scatter_nd_sub_1"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid, float32
    ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    indices = np.array([[0], [2], [4]], dtype=np.int32)
    updates = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    use_locking = True
    bad_indices_policy = "ignore"
    name = "scatter_nd_sub_2"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid, int64 indices
    ref = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    indices = np.array([[0], [2], [4]], dtype=np.int64)
    updates = np.array([1, 2, 3], dtype=np.int32)
    use_locking = False
    bad_indices_policy = ""
    name = "scatter_nd_sub_3"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid, multi-dimensional
    ref = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    indices = np.array([[0], [2]], dtype=np.int32)
    updates = np.array([[1, 1], [2, 2]], dtype=np.int32)
    use_locking = True
    bad_indices_policy = "ignore"
    name = "scatter_nd_sub_4"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid, K < P
    ref = np.arange(24, dtype=np.int32).reshape((2, 3, 4))
    indices = np.array([[0, 0], [1, 2]], dtype=np.int32)
    updates = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)
    use_locking = False
    bad_indices_policy = ""
    name = "scatter_nd_sub_5"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6, valid, K < P, different shape
    ref = np.arange(24, dtype=np.int32).reshape((2, 3, 4))
    indices = np.array([[0,0,0], [1,1,1]], dtype=np.int32) #modified indices
    updates = np.array([1,2],dtype=np.int32) #Shape update. Made it a valid shape

    use_locking = False
    bad_indices_policy = ""
    name = "scatter_nd_sub_6"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid, negative updates
    ref = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    indices = np.array([[0], [2], [4]], dtype=np.int32)
    updates = np.array([-1, -2, -3], dtype=np.int32)
    use_locking = True
    bad_indices_policy = "ignore"
    name = "scatter_nd_sub_7"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid, different data type
    ref = np.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype=np.float64)
    indices = np.array([[0], [2], [4]], dtype=np.int32)
    updates = np.array([0.5, 1.5, 2.5], dtype=np.float64)
    use_locking = False
    bad_indices_policy = ""
    name = "scatter_nd_sub_8"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid, empty indices
    ref = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    indices = np.array([], dtype=np.int32).reshape(0,1)
    updates = np.array([], dtype=np.int32)
    use_locking = True
    bad_indices_policy = "ignore"
    name = "scatter_nd_sub_9"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid, uint8
    ref = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
    indices = np.array([[0], [2], [4]], dtype=np.int32)
    updates = np.array([1, 2, 3], dtype=np.uint8)
    use_locking = False
    bad_indices_policy = ""
    name = "scatter_nd_sub_10"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ScatterNdSub"] = tf_raw_ops_scatter_nd_sub_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ScatterNdSub' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ScatterNdSub'.")

check_valid('tf.raw_ops.ScatterNdSub', generated_inputs['tf.raw_ops.ScatterNdSub'], lib="tf", suffix=0)
