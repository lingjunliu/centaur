
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ScatterNdSub_inputs():
    list_of_inputs = []

    # Input 1
    ref = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    indices = np.array([[0], [2], [4]], dtype=np.int32)
    updates = np.array([10, 20, 30], dtype=np.int32)
    use_locking = False
    bad_indices_policy = ""
    name = "test_scatter_nd_sub_1"

    input_dict = {
        "ref": ref,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "bad_indices_policy": bad_indices_policy,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    ref = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    indices = np.array([[0], [1]], dtype=np.int64)
    updates = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    use_locking = True
    bad_indices_policy = ""
    name = "test_scatter_nd_sub_2"

    input_dict = {
        "ref": ref,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "bad_indices_policy": bad_indices_policy,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    ref = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    indices = np.array([[0, 0], [1, 1]], dtype=np.int32)
    updates = np.array([[10, 20], [30, 40]], dtype=np.int64)
    use_locking = False
    bad_indices_policy = ""
    name = "test_scatter_nd_sub_3"

    input_dict = {
        "ref": ref,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "bad_indices_policy": bad_indices_policy,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    ref = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    indices = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int64)
    updates = np.array([1, 2, 3], dtype=np.int32)
    use_locking = True
    bad_indices_policy = ""
    name = "test_scatter_nd_sub_4"

    input_dict = {
        "ref": ref,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "bad_indices_policy": bad_indices_policy,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 5
    ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    indices = np.array([[0], [2], [4]], dtype=np.int32)
    updates = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    use_locking = False
    bad_indices_policy = ""
    name = "test_scatter_nd_sub_5"

    input_dict = {
        "ref": ref,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "bad_indices_policy": bad_indices_policy,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    ref = np.array([1, 2, 3, 4], dtype=np.int32)
    indices = np.array([[1], [3]], dtype=np.int32)
    updates = np.array([-5, -10], dtype=np.int32)
    use_locking = False
    bad_indices_policy = ""
    name = "test_scatter_nd_sub_6"

    input_dict = {
        "ref": ref,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "bad_indices_policy": bad_indices_policy,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: multi-dimensional ref
    ref = np.array([[1, 2], [3, 4]], dtype=np.int32)
    indices = np.array([[0, 0], [1, 1]], dtype=np.int32)
    updates = np.array([5, 10], dtype=np.int32)
    use_locking = False
    bad_indices_policy = ""
    name = "test_scatter_nd_sub_7"

    input_dict = {
        "ref": ref,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "bad_indices_policy": bad_indices_policy,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    ref = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    indices = np.array([[0], [2], [4]], dtype=np.int64)
    updates = np.array([10, 20, 30], dtype=np.int64)
    use_locking = False
    bad_indices_policy = ""
    name = "test_scatter_nd_sub_8"

    input_dict = {
        "ref": ref,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "bad_indices_policy": bad_indices_policy,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty updates
    ref = np.array([1, 2, 3], dtype=np.int32)
    indices = np.array([], dtype=np.int32).reshape(0, 1)
    updates = np.array([], dtype=np.int32)
    use_locking = False
    bad_indices_policy = ""
    name = "test_scatter_nd_sub_9"

    input_dict = {
        "ref": ref,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "bad_indices_policy": bad_indices_policy,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Large indices
    ref = np.zeros((100,), dtype=np.int32)
    indices = np.array([[i] for i in range(100)], dtype=np.int32)
    updates = np.ones((100,), dtype=np.int32)
    use_locking = False
    bad_indices_policy = ""
    name = "test_scatter_nd_sub_10"

    input_dict = {
        "ref": ref,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "bad_indices_policy": bad_indices_policy,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: rank 3 ref, rank 2 indices
    ref = np.arange(24).reshape((2, 3, 4)).astype(np.int32)
    indices = np.array([[0, 1], [1, 2]], dtype=np.int32)
    updates = np.array([[100, 101, 102, 103], [200, 201, 202, 203]], dtype=np.int32)
    use_locking = False
    bad_indices_policy = ""
    name = "test_scatter_nd_sub_11"

    input_dict = {
        "ref": ref,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "bad_indices_policy": bad_indices_policy,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.ScatterNdSub"] = tf_raw_ops_ScatterNdSub_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ScatterNdSub' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ScatterNdSub'.")

check_valid('tf.raw_ops.ScatterNdSub', generated_inputs['tf.raw_ops.ScatterNdSub'], lib="tf", suffix=0)
