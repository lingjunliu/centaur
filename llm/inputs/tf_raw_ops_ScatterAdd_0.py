
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ScatterAdd_inputs():
    list_of_inputs = []

    # Input 1
    ref = np.array([1, 2, 3, 4], dtype=np.int32)
    indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([5, 6], dtype=np.int32)
    use_locking = False
    name = "scatter_add_1"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    ref = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    use_locking = True
    name = "scatter_add_2"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    ref = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    indices = np.array([0, 1, 2], dtype=np.int32)
    updates = np.array([0.5, 1.0, 1.5], dtype=np.float64)
    use_locking = False
    name = "scatter_add_3"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    ref = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64)
    indices = np.array([0, 1], dtype=np.int64)
    updates = np.array([[10, 11, 12], [13, 14, 15]], dtype=np.int64)
    use_locking = True
    name = "scatter_add_4"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5
    ref = np.array([1, 2, 3, 4], dtype=np.int32)
    indices = np.array([0, 1, 2, 3], dtype=np.int32)
    updates = np.array([5, 6, 7, 8], dtype=np.int32)
    use_locking = False
    name = "scatter_add_5"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    ref = np.array([[1, 2], [3, 4]], dtype=np.float32)
    indices = np.array([1, 0], dtype=np.int32)
    updates = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    use_locking = True
    name = "scatter_add_6"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    ref = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
    indices = np.array([5, 0, 2], dtype=np.int32)
    updates = np.array([7, 8, 9], dtype=np.int32)
    use_locking = False
    name = "scatter_add_7"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    ref = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    updates = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    use_locking = True
    name = "scatter_add_8"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    indices = np.array([4, 3, 2, 1, 0], dtype=np.int32)
    updates = np.array([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float64)
    use_locking = False
    name = "scatter_add_9"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    ref = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.int64)
    indices = np.array([3, 0], dtype=np.int64)
    updates = np.array([[10, 11], [13, 14]], dtype=np.int64)
    use_locking = True
    name = "scatter_add_10"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ScatterAdd"] = tf_raw_ops_ScatterAdd_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ScatterAdd' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ScatterAdd'.")

check_valid('tf.raw_ops.ScatterAdd', generated_inputs['tf.raw_ops.ScatterAdd'], lib="tf", suffix=0)
