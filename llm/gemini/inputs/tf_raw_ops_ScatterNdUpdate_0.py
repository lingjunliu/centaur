
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_ScatterNdUpdate_inputs():
    list_of_inputs = []

    # Input 1: Basic example
    ref = tf.Variable(np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32))
    indices = np.array([[4], [3], [1], [7]], dtype=np.int32)
    updates = np.array([9, 10, 11, 12], dtype=np.int32)
    use_locking = True
    bad_indices_policy = ""
    name = "basic_example"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D ref, 2D indices
    ref = tf.Variable(np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32))
    indices = np.array([[0, 0], [1, 1]], dtype=np.int32)
    updates = np.array([7, 8], dtype=np.int32)
    use_locking = False
    bad_indices_policy = ""
    name = "2d_example"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D ref, 2D indices
    ref = tf.Variable(np.arange(24, dtype=np.int32).reshape((2, 3, 4)))
    indices = np.array([[0, 1], [1, 2]], dtype=np.int32)
    updates = np.array([[100, 101, 102, 103], [200, 201, 202, 203]], dtype=np.int32)
    use_locking = True
    bad_indices_policy = ""
    name = "3d_example"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int64 indices
    ref = tf.Variable(np.array([1, 2, 3, 4, 5], dtype=np.int32))
    indices = np.array([[0], [2], [4]], dtype=np.int64)
    updates = np.array([10, 20, 30], dtype=np.int32)
    use_locking = True
    bad_indices_policy = ""
    name = "int64_indices"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different data type for ref and updates (float32)
    ref = tf.Variable(np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32))
    indices = np.array([[0], [2], [4]], dtype=np.int32)
    updates = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    use_locking = True
    bad_indices_policy = ""
    name = "float32_example"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ScatterNdUpdate"] = tf_raw_ops_ScatterNdUpdate_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ScatterNdUpdate' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ScatterNdUpdate'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.ScatterNdUpdate', generated_inputs['tf.raw_ops.ScatterNdUpdate'], lib="tf", suffix=0)
