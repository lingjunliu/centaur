
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_scatter_update_inputs():
    list_of_inputs = []

    # Input 1: Basic example
    ref = np.array([1, 2, 3, 4, 5]).astype(np.int32)
    indices = np.array([0, 2, 4]).astype(np.int32)
    updates = np.array([10, 20, 30]).astype(np.int32)
    use_locking = True
    name = "scatter_update_1"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different data type for ref and updates
    ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).astype(np.float32)
    indices = np.array([1, 3]).astype(np.int32)
    updates = np.array([6.0, 7.0]).astype(np.float32)
    use_locking = False
    name = "scatter_update_2"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multi-dimensional ref
    ref = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.int32)
    indices = np.array([0, 2]).astype(np.int32)
    updates = np.array([[10, 20], [50, 60]]).astype(np.int32)
    use_locking = True
    name = "scatter_update_3"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty updates
    ref = np.array([1, 2, 3]).astype(np.int32)
    indices = np.array([]).astype(np.int32)
    updates = np.array([]).astype(np.int32)
    use_locking = False
    name = "scatter_update_4"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Zero index
    ref = np.array([1, 2, 3]).astype(np.int32)
    indices = np.array([0]).astype(np.int32)
    updates = np.array([10]).astype(np.int32)
    use_locking = True
    name = "scatter_update_5"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative values
    ref = np.array([-1, -2, -3]).astype(np.int32)
    indices = np.array([0, 1]).astype(np.int32)
    updates = np.array([-10, -20]).astype(np.int32)
    use_locking = False
    name = "scatter_update_6"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7: Float64
    ref = np.array([1.0, 2.0, 3.0]).astype(np.float64)
    indices = np.array([0, 1]).astype(np.int32)
    updates = np.array([4.0, 5.0]).astype(np.float64)
    use_locking = True
    name = "scatter_update_7"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Int64 indices
    ref = np.array([1, 2, 3]).astype(np.int32)
    indices = np.array([0, 1]).astype(np.int64)
    updates = np.array([4, 5]).astype(np.int32)
    use_locking = False
    name = "scatter_update_8"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Updates shape is empty, float32
    ref = np.array([1.0, 2.0, 3.0]).astype(np.float32)
    indices = np.array([1]).astype(np.int32)
    updates = np.array(100.0).astype(np.float32)
    use_locking = True
    name = "scatter_update_9"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger indices, different dtype for indices, multi dim updates
    ref = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.int32)
    indices = np.array([2, 5]).astype(np.int64)
    updates = np.array([22, 55]).astype(np.int32)
    use_locking = False
    name = "scatter_update_10"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ScatterUpdate"] = tf_raw_ops_scatter_update_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ScatterUpdate' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ScatterUpdate'.")

check_valid('tf.raw_ops.ScatterUpdate', generated_inputs['tf.raw_ops.ScatterUpdate'], lib="tf", suffix=0)
