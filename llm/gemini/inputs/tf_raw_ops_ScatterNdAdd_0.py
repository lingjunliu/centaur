
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ScatterNdAdd_inputs():
    list_of_inputs = []

    # Input 1
    ref = tf.Variable(np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32), aggregation=tf.VariableAggregation.SUM)
    indices = np.array([[4], [3], [1], [7]], dtype=np.int32)
    updates = np.array([9, 10, 11, 12], dtype=np.int32)
    use_locking = False
    bad_indices_policy = ""
    name = "scatter_add_1"

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
    ref = tf.Variable(np.array([[1, 2], [3, 4]], dtype=np.float32), aggregation=tf.VariableAggregation.SUM)
    indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    updates = np.array([5.0, 6.0], dtype=np.float32)
    use_locking = True
    bad_indices_policy = "ignore"
    name = "scatter_add_2"

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
    ref = tf.Variable(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64), aggregation=tf.VariableAggregation.SUM)
    indices = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int32)
    updates = np.array([9, 10], dtype=np.int64)
    use_locking = False
    bad_indices_policy = "warn"
    name = "scatter_add_3"

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
    ref = tf.Variable(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64), aggregation=tf.VariableAggregation.SUM)
    indices = np.array([[0], [2]], dtype=np.int64)
    updates = np.array([[10, 11, 12], [13, 14, 15]], dtype=np.float64)
    use_locking = True
    bad_indices_policy = ""
    name = "scatter_add_4"

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
    ref = tf.Variable(np.array([1, 2, 3], dtype=np.int32), aggregation=tf.VariableAggregation.SUM)
    indices = np.array([[0], [1], [2]], dtype=np.int32)
    updates = np.array([4, 5, 6], dtype=np.int32)
    use_locking = False
    bad_indices_policy = ""
    name = None

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
    ref = tf.Variable(np.array([[1, 2], [3, 4]], dtype=np.int32), aggregation=tf.VariableAggregation.SUM)
    indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int32)
    updates = np.array([5, 6, 7, 8], dtype=np.int32)
    use_locking = True
    bad_indices_policy = ""
    name = "scatter_add_6"

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

generated_inputs = {}
generated_inputs["tf.raw_ops.ScatterNdAdd"] = tf_raw_ops_ScatterNdAdd_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ScatterNdAdd' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ScatterNdAdd'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.ScatterNdAdd', generated_inputs['tf.raw_ops.ScatterNdAdd'], lib="tf", suffix=0)
