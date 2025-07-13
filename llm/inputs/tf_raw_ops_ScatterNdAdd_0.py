
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_scatter_nd_add_inputs():
    list_of_inputs = []

    # Input 1
    ref = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
    indices = np.array([[4], [3], [1], [7]], dtype=np.int32)
    updates = np.array([9, 10, 11, 12], dtype=np.int32)
    use_locking = False
    bad_indices_policy = ""
    name = None

    input_dict = {
        "ref": tf.Variable(ref).value(),
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "bad_indices_policy": bad_indices_policy,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    ref = np.array([[1, 2], [3, 4]], dtype=np.float32)
    indices = np.array([[0, 0], [1, 1]], dtype=np.int32)
    updates = np.array([5, 6], dtype=np.float32)
    use_locking = True
    bad_indices_policy = ""
    name = "scatter_add_op"

    input_dict = {
        "ref": tf.Variable(ref).value(),
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "bad_indices_policy": bad_indices_policy,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    ref = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float64)
    indices = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int32)
    updates = np.array([9, 10], dtype=np.float64)
    use_locking = False
    bad_indices_policy = ""
    name = None

    input_dict = {
        "ref": tf.Variable(ref).value(),
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "bad_indices_policy": bad_indices_policy,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    ref = np.array([1, 2, 3, 4], dtype=np.int64)
    indices = np.array([[0], [2]], dtype=np.int64)
    updates = np.array([5, 6], dtype=np.int64)
    use_locking = True
    bad_indices_policy = ""
    name = "test"

    input_dict = {
        "ref": tf.Variable(ref).value(),
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "bad_indices_policy": bad_indices_policy,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    ref = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.complex64)
    indices = np.array([[0, 1], [1, 2]], dtype=np.int32)
    updates = np.array([7 + 1j, 8 + 2j], dtype=np.complex64)
    use_locking = False
    bad_indices_policy = ""
    name = None

    input_dict = {
        "ref": tf.Variable(ref).value(),
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "bad_indices_policy": bad_indices_policy,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    ref = np.array([1, 2, 3, 4], dtype=np.float32)
    indices = np.array([[0], [1], [2], [3]], dtype=np.int32)
    updates = np.array([-1.0, -2.0, -3.0, -4.0], dtype=np.float32)
    use_locking = True
    bad_indices_policy = ""
    name = "neg_update"

    input_dict = {
        "ref": tf.Variable(ref).value(),
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "bad_indices_policy": bad_indices_policy,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: multi dimensional updates
    ref = np.zeros((5, 5), dtype=np.float32)
    indices = np.array([[1], [3]], dtype=np.int32)
    updates = np.ones((2, 5), dtype=np.float32)
    use_locking = False
    bad_indices_policy = ""
    name = None

    input_dict = {
        "ref": tf.Variable(ref).value(),
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "bad_indices_policy": bad_indices_policy,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: higher rank ref and indices
    ref = np.zeros((2, 3, 4), dtype=np.int32)
    indices = np.array([[0, 1], [1, 2]], dtype=np.int32)
    updates = np.ones((2, 4), dtype=np.int32)
    use_locking = True
    bad_indices_policy = ""
    name = "rank_3"

    input_dict = {
        "ref": tf.Variable(ref).value(),
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "bad_indices_policy": bad_indices_policy,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9: uint32
    ref = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint32)
    indices = np.array([[4], [3], [1], [7]], dtype=np.int32)
    updates = np.array([9, 10, 11, 12], dtype=np.uint32)
    use_locking = False
    bad_indices_policy = ""
    name = None

    input_dict = {
        "ref": tf.Variable(ref).value(),
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "bad_indices_policy": bad_indices_policy,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: uint64
    ref = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint64)
    indices = np.array([[4], [3], [1], [7]], dtype=np.int32)
    updates = np.array([9, 10, 11, 12], dtype=np.uint64)
    use_locking = False
    bad_indices_policy = ""
    name = None

    input_dict = {
        "ref": tf.Variable(ref).value(),
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "bad_indices_policy": bad_indices_policy,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ScatterNdAdd"] = tf_raw_ops_scatter_nd_add_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ScatterNdAdd' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ScatterNdAdd'.")

check_valid('tf.raw_ops.ScatterNdAdd', generated_inputs['tf.raw_ops.ScatterNdAdd'], lib="tf", suffix=0)
