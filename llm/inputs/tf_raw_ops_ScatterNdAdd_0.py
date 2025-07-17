
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_scatter_nd_add_inputs():
    list_of_inputs = []

    # Input 1: Basic example, rank-1 ref
    ref = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
    indices = np.array([[4], [3], [1], [7]], dtype=np.int32)
    updates = np.array([9, 10, 11, 12], dtype=np.int32)
    use_locking = False
    bad_indices_policy = ""
    name = "scatter_add_1"
    input_dict = {"ref": tf.Variable(ref).value(), "indices": indices, "updates": updates, "use_locking": use_locking, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Rank-2 ref, K=1
    ref = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    indices = np.array([[0], [2]], dtype=np.int32)
    updates = np.array([[10, 20], [50, 60]], dtype=np.float32)
    use_locking = True
    bad_indices_policy = "ignore"
    name = "scatter_add_2"
    input_dict = {"ref": tf.Variable(ref).value(), "indices": indices, "updates": updates, "use_locking": use_locking, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Rank-2 ref, K=2
    ref = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64)
    indices = np.array([[0, 0], [1, 1], [2, 0]], dtype=np.int64)
    updates = np.array([10, 40, 50], dtype=np.int64)
    use_locking = False
    bad_indices_policy = "warn"
    name = "scatter_add_3"
    input_dict = {"ref": tf.Variable(ref).value(), "indices": indices, "updates": updates, "use_locking": use_locking, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Rank-3 ref, K=1
    ref = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float64)
    indices = np.array([[0], [1]], dtype=np.int32)
    updates = np.array([[[10, 20], [30, 40]], [[50, 60], [70, 80]]], dtype=np.float64)
    use_locking = True
    bad_indices_policy = ""
    name = "scatter_add_4"
    input_dict = {"ref": tf.Variable(ref).value(), "indices": indices, "updates": updates, "use_locking": use_locking, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Rank-3 ref, K=2
    ref = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    indices = np.array([[0, 0], [1, 1]], dtype=np.int32)
    updates = np.array([[10, 20], [70, 80]], dtype=np.int32)
    use_locking = False
    bad_indices_policy = ""
    name = "scatter_add_5"
    input_dict = {"ref": tf.Variable(ref).value(), "indices": indices, "updates": updates, "use_locking": use_locking, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Rank-3 ref, K=3
    ref = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    indices = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int64)
    updates = np.array([10, 80], dtype=np.float32)
    use_locking = True
    bad_indices_policy = ""
    name = "scatter_add_6"
    input_dict = {"ref": tf.Variable(ref).value(), "indices": indices, "updates": updates, "use_locking": use_locking, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Rank-4 ref, K=2
    ref = np.arange(1, 17, dtype=np.int32).reshape((2, 2, 2, 2))
    indices = np.array([[0, 0], [1, 1]], dtype=np.int32)
    updates = np.arange(10, 26, dtype=np.int32).reshape((2, 2, 2, 2))
    use_locking = False
    bad_indices_policy = ""
    name = "scatter_add_7"
    input_dict = {"ref": tf.Variable(ref).value(), "indices": indices, "updates": updates, "use_locking": use_locking, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Rank-2 ref, K=1, int64 indices
    ref = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    indices = np.array([[0], [2]], dtype=np.int64)
    updates = np.array([[10, 20], [50, 60]], dtype=np.int32)
    use_locking = True
    bad_indices_policy = ""
    name = "scatter_add_8"
    input_dict = {"ref": tf.Variable(ref).value(), "indices": indices, "updates": updates, "use_locking": use_locking, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Rank-1 ref, negative updates
    ref = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
    indices = np.array([[4], [3], [1], [7]], dtype=np.int32)
    updates = np.array([-9, -10, -11, -12], dtype=np.int32)
    use_locking = False
    bad_indices_policy = ""
    name = "scatter_add_9"
    input_dict = {"ref": tf.Variable(ref).value(), "indices": indices, "updates": updates, "use_locking": use_locking, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 10: Rank-2 ref, K=2, int64 ref and updates
    ref = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64)
    indices = np.array([[0, 0], [1, 1], [2, 0]], dtype=np.int64)
    updates = np.array([10, 40, 50], dtype=np.int64)
    use_locking = False
    bad_indices_policy = ""
    name = "scatter_add_10"
    input_dict = {"ref": tf.Variable(ref).value(), "indices": indices, "updates": updates, "use_locking": use_locking, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
try:
    tf.config.experimental_run_functions_eagerly(False)
except:
    pass
generated_inputs["tf.raw_ops.ScatterNdAdd"] = tf_raw_ops_scatter_nd_add_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ScatterNdAdd' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ScatterNdAdd'.")

check_valid('tf.raw_ops.ScatterNdAdd', generated_inputs['tf.raw_ops.ScatterNdAdd'], lib="tf", suffix=0)
