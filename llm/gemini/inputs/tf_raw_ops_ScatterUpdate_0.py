
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_scatter_update_inputs():
    list_of_inputs = []

    # Input 1
    ref = tf.Variable(np.array([1, 2, 3, 4, 5], dtype=np.int32)).ref()
    indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([10, 20], dtype=np.int32)
    use_locking = True
    name = "scatter_update_1"

    input_dict = {
        "ref": ref,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    ref = tf.Variable(np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)).ref()
    indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([[10, 20], [50, 60]], dtype=np.float32)
    use_locking = False
    name = "scatter_update_2"

    input_dict = {
        "ref": ref,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    ref = tf.Variable(np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)).ref()
    indices = np.array([1, 3], dtype=np.int64)
    updates = np.array([10.0, 20.0], dtype=np.float64)
    use_locking = True
    name = "scatter_update_3"

    input_dict = {
        "ref": ref,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    ref = tf.Variable(np.array([1, 2, 3, 4, 5], dtype=np.int64)).ref()
    indices = np.array([0, 4, 2], dtype=np.int32)
    updates = np.array([-10, -20, -30], dtype=np.int64)
    use_locking = False
    name = "scatter_update_4"

    input_dict = {
        "ref": ref,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    ref = tf.Variable(np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)).ref()
    indices = np.array([0], dtype=np.int32)
    updates = np.array([[10, 20]], dtype=np.int32)
    use_locking = True
    name = "scatter_update_5"

    input_dict = {
        "ref": ref,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    ref = tf.Variable(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)).ref()
    indices = np.array([0, 1], dtype=np.int32)
    updates = np.array([[[10, 20], [30, 40]], [[50, 60], [70, 80]]], dtype=np.float32)
    use_locking = False
    name = "scatter_update_6"

    input_dict = {
        "ref": ref,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    ref = tf.Variable(np.array([1, 2, 3, 4, 5], dtype=np.float32)).ref()
    indices = np.array([], dtype=np.int32)
    updates = np.array([], dtype=np.float32)
    use_locking = True
    name = "scatter_update_7"

    input_dict = {
        "ref": ref,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    ref = tf.Variable(np.array([[1, 2], [3, 4]], dtype=np.int64)).ref()
    indices = np.array([0, 1], dtype=np.int64)
    updates = np.array([[5, 6], [7, 8]], dtype=np.int64)
    use_locking = False
    name = "scatter_update_8"

    input_dict = {
        "ref": ref,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9
    ref = tf.Variable(np.array([1, 2, 3], dtype=np.int32)).ref()
    indices = np.array([2,0], dtype=np.int32)
    updates = np.array([7,8], dtype=np.int32)
    use_locking = True
    name = "scatter_update_9"

    input_dict = {
        "ref": ref,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    ref = tf.Variable(np.array([1.0, 2.0, 3.0], dtype=np.float32)).ref()
    indices = np.array([0, 1, 2], dtype=np.int32)
    updates = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    use_locking = False
    name = "scatter_update_10"

    input_dict = {
        "ref": ref,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ScatterUpdate"] = tf_raw_ops_scatter_update_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ScatterUpdate' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ScatterUpdate'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.ScatterUpdate', generated_inputs['tf.raw_ops.ScatterUpdate'], lib="tf", suffix=0)
