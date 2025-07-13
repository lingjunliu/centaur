
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ScatterMin_inputs():
    list_of_inputs = []

    # Input 1
    ref = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([0.5, 1.5], dtype=np.float32)
    use_locking = False
    name = "scatter_min_1"

    input_dict = {
        "ref": ref,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    ref = np.array([[1, 2], [3, 4]], dtype=np.int32)
    indices = np.array([0], dtype=np.int32)
    updates = np.array([[0, 5]], dtype=np.int32)
    use_locking = True
    name = "scatter_min_2"
    input_dict = {
        "ref": ref,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    ref = np.array([5, 6, 7, 8], dtype=np.int64)
    indices = np.array([1, 3], dtype=np.int64)
    updates = np.array([1, 2], dtype=np.int64)
    use_locking = False
    name = "scatter_min_3"
    input_dict = {
        "ref": ref,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    ref = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([-0.5, -1.5], dtype=np.float64)
    use_locking = True
    name = "scatter_min_4"
    input_dict = {
        "ref": ref,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    ref = np.array([[1, 2], [3, 4]], dtype=np.int32)
    indices = np.array([0, 1], dtype=np.int32)
    updates = np.array([[0, 1], [2, 0]], dtype=np.int32)
    use_locking = False
    name = "scatter_min_5"
    input_dict = {
        "ref": ref,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    ref = np.array([5, 6, 7, 8], dtype=np.int64)
    indices = np.array([1, 3, 1, 3], dtype=np.int64)
    updates = np.array([1, 2, 0, -1], dtype=np.int64)
    use_locking = True
    name = "scatter_min_6"
    input_dict = {
        "ref": ref,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    ref = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    indices = np.array([0, 1, 2, 3], dtype=np.int32)
    updates = np.array([0.0, -1.0, 2.0, -3.0], dtype=np.float32)
    use_locking = False
    name = "scatter_min_7"
    input_dict = {
        "ref": ref,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8
    ref = np.array([1, 2, 3, 4], dtype=np.int32)
    indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([0, 1], dtype=np.int32)
    use_locking = True
    name = "scatter_min_8"
    input_dict = {
        "ref": ref,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    ref = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    indices = np.array([[0], [1]], dtype=np.int32)
    updates = np.array([[0.5, 1.5], [2.5, 0.5]], dtype=np.float32)
    use_locking = False
    name = "scatter_min_9"
    input_dict = {
        "ref": ref,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    ref = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    indices = np.array([0,1,2], dtype=np.int32)
    updates = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    use_locking = True
    name = "scatter_min_10"
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
generated_inputs["tf.raw_ops.ScatterMin"] = tf_raw_ops_ScatterMin_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ScatterMin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ScatterMin'.")

check_valid('tf.raw_ops.ScatterMin', generated_inputs['tf.raw_ops.ScatterMin'], lib="tf", suffix=0)
