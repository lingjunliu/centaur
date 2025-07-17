
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_scatter_update_inputs():
    list_of_inputs = []

    def to_numpy(tensor):
        return tensor.numpy()

    # Input 1: Basic valid case
    ref = tf.Variable(np.array([1, 2, 3, 4, 5], dtype=np.float32))
    indices = np.array([0, 2, 4], dtype=np.int32)
    updates = np.array([7, 8, 9], dtype=np.float32)
    use_locking = True
    name = "scatter_update_1"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "name": name}
    input_dict['ref'] = to_numpy(input_dict['ref'])
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Multi-dimensional ref
    ref = tf.Variable(np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32))
    indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([[7, 8], [9, 10]], dtype=np.float32)
    use_locking = False
    name = "scatter_update_2"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "name": name}
    input_dict['ref'] = to_numpy(input_dict['ref'])
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different data type
    ref = tf.Variable(np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32))
    indices = np.array([1, 3], dtype=np.int32)
    updates = np.array([7.0, 8.0], dtype=np.float32)
    use_locking = True
    name = "scatter_update_3"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "name": name}
    input_dict['ref'] = to_numpy(input_dict['ref'])
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different indices data type
    ref = tf.Variable(np.array([1, 2, 3, 4, 5], dtype=np.float32))
    indices = np.array([0, 2, 4], dtype=np.int64)
    updates = np.array([7, 8, 9], dtype=np.float32)
    use_locking = False
    name = "scatter_update_4"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "name": name}
    input_dict['ref'] = to_numpy(input_dict['ref'])
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5: Updates shape is []
    ref = tf.Variable(np.array([1, 2, 3, 4, 5], dtype=np.float32))
    indices = np.array([0], dtype=np.int32)
    updates = np.array(7, dtype=np.float32)
    use_locking = True
    name = "scatter_update_5"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "name": name}
    input_dict['ref'] = to_numpy(input_dict['ref'])
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Another multi-dimensional example
    ref = tf.Variable(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32))
    indices = np.array([0, 1], dtype=np.int32)
    updates = np.array([[10, 11, 12], [13, 14, 15]], dtype=np.float32)
    use_locking = False
    name = "scatter_update_6"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "name": name}
    input_dict['ref'] = to_numpy(input_dict['ref'])
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Negative values
    ref = tf.Variable(np.array([-1, -2, -3, -4, -5], dtype=np.float32))
    indices = np.array([0, 2, 4], dtype=np.int32)
    updates = np.array([-7, -8, -9], dtype=np.float32)
    use_locking = True
    name = "scatter_update_7"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "name": name}
    input_dict['ref'] = to_numpy(input_dict['ref'])
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Zero values
    ref = tf.Variable(np.array([0, 0, 0, 0, 0], dtype=np.float32))
    indices = np.array([1, 3], dtype=np.int32)
    updates = np.array([7, 8], dtype=np.float32)
    use_locking = False
    name = "scatter_update_8"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "name": name}
    input_dict['ref'] = to_numpy(input_dict['ref'])
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Larger values
    ref = tf.Variable(np.array([1000, 2000, 3000], dtype=np.float32))
    indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([7000, 9000], dtype=np.float32)
    use_locking = True
    name = "scatter_update_9"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "name": name}
    input_dict['ref'] = to_numpy(input_dict['ref'])
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: More complex ref
    ref = tf.Variable(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32))
    indices = np.array([0], dtype=np.int32)
    updates = np.array([[[9, 10], [11, 12]]], dtype=np.float32)
    use_locking = False
    name = "scatter_update_10"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "name": name}
    input_dict['ref'] = to_numpy(input_dict['ref'])
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
