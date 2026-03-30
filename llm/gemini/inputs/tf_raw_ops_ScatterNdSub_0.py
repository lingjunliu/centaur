
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_scatter_nd_sub_inputs():
    list_of_inputs = []

    # Input 1
    ref = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
    indices = np.array([[4], [3], [1], [7]], dtype=np.int32)
    updates = np.array([9, 10, 11, 12], dtype=np.int32)
    use_locking = False
    bad_indices_policy = ""
    name = None

    input_dict = {
        "ref": tf.Variable(ref, use_resource=True).ref(),
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "bad_indices_policy": bad_indices_policy,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    ref = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    indices = np.array([[0], [2]], dtype=np.int32)
    updates = np.array([[10, 10], [20, 20]], dtype=np.float32)
    use_locking = True
    bad_indices_policy = "ignore"
    name = "scatter_sub_op"

    input_dict = {
        "ref": tf.Variable(ref, use_resource=True).ref(),
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
    updates = np.array([[10, 10], [20, 20]], dtype=np.int64)
    use_locking = False
    bad_indices_policy = ""
    name = None

    input_dict = {
        "ref": tf.Variable(ref, use_resource=True).ref(),
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "bad_indices_policy": bad_indices_policy,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    ref = np.array([1, 2, 3, 4], dtype=np.float64)
    indices = np.array([[0], [1], [2]], dtype=np.int64)
    updates = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    use_locking = True
    bad_indices_policy = ""
    name = "sub_op"

    input_dict = {
        "ref": tf.Variable(ref, use_resource=True).ref(),
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "bad_indices_policy": bad_indices_policy,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    ref = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    indices = np.array([[0, 0], [0, 1], [1, 2]], dtype=np.int32)
    updates = np.array([-1, -2, -3], dtype=np.int32)
    use_locking = False
    bad_indices_policy = ""
    name = None

    input_dict = {
        "ref": tf.Variable(ref, use_resource=True).ref(),
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "bad_indices_policy": bad_indices_policy,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    ref = np.array([1, 2, 3], dtype=np.float32)
    indices = np.array([[0], [1]], dtype=np.int32)
    updates = np.array([0.5, 1.5], dtype=np.float32)
    use_locking = False
    bad_indices_policy = ""
    name = None
    input_dict = {
        "ref": tf.Variable(ref, use_resource=True).ref(),
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "bad_indices_policy": bad_indices_policy,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    ref = np.array([[1, 2], [3, 4]], dtype=np.int64)
    indices = np.array([[0, 0], [1, 1]], dtype=np.int32)
    updates = np.array([1, 2], dtype=np.int64)
    use_locking = True
    bad_indices_policy = "warn"
    name = "test_op"
    input_dict = {
        "ref": tf.Variable(ref, use_resource=True).ref(),
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "bad_indices_policy": bad_indices_policy,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    ref = np.array([1, 2, 3, 4], dtype=np.float64)
    indices = np.array([[0], [2], [3]], dtype=np.int32)
    updates = np.array([0.1, 0.3, 0.4], dtype=np.float64)
    use_locking = False
    bad_indices_policy = ""
    name = None
    input_dict = {
        "ref": tf.Variable(ref, use_resource=True).ref(),
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "bad_indices_policy": bad_indices_policy,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    ref = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    indices = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int32)
    updates = np.array([1, 2], dtype=np.int32)
    use_locking = True
    bad_indices_policy = ""
    name = "test_op"
    input_dict = {
        "ref": tf.Variable(ref, use_resource=True).ref(),
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "bad_indices_policy": bad_indices_policy,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    ref = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    indices = np.array([[0, 0], [0, 1], [1, 1]], dtype=np.int32)
    updates = np.array([1, 2, 3], dtype=np.int32)
    use_locking = False
    bad_indices_policy = ""
    name = None
    input_dict = {
        "ref": tf.Variable(ref, use_resource=True).ref(),
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "bad_indices_policy": bad_indices_policy,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.ScatterNdSub"] = tf_raw_ops_scatter_nd_sub_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ScatterNdSub' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ScatterNdSub'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.ScatterNdSub', generated_inputs['tf.raw_ops.ScatterNdSub'], lib="tf", suffix=0)
