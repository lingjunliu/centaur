
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_gather_inputs():
    list_of_inputs = []

    params1 = np.array([[1, 2], [3, 4], [5, 6]])
    indices1 = np.array([0, 1, 2], dtype=np.int32)
    validate_indices1 = True
    name1 = "gather_test_1"

    input_dict1 = {
        "params": params1,
        "indices": indices1,
        "validate_indices": validate_indices1,
        "name": name1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    params2 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    indices2 = np.array([[0, 1], [1, 0]], dtype=np.int64)
    validate_indices2 = False
    name2 = "gather_test_2"

    input_dict2 = {
        "params": params2,
        "indices": indices2,
        "validate_indices": validate_indices2,
        "name": name2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    params3 = np.array([10, 20, 30, 40])
    indices3 = np.array([2, 0, 1], dtype=np.int32)
    validate_indices3 = True
    name3 = "gather_test_3"

    input_dict3 = {
        "params": params3,
        "indices": indices3,
        "validate_indices": validate_indices3,
        "name": name3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    params4 = np.array([[1, 2, 3], [4, 5, 6]])
    indices4 = np.array([0, 0, 1], dtype=np.int64)
    validate_indices4 = False
    name4 = "gather_test_4"

    input_dict4 = {
        "params": params4,
        "indices": indices4,
        "validate_indices": validate_indices4,
        "name": name4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    params5 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    indices5 = np.array([[[0, 0], [0, 1]], [[1, 0], [1, 1]]], dtype=np.int64)
    validate_indices5 = True
    name5 = "gather_test_5"

    input_dict5 = {
        "params": params5,
        "indices": indices5,
        "validate_indices": validate_indices5,
        "name": name5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    params6 = np.array([[1, 2], [3, 4]])
    indices6 = np.array([0, 1], dtype=np.int32)
    validate_indices6 = False
    name6 = "gather_test_6"

    input_dict6 = {
        "params": params6,
        "indices": indices6,
        "validate_indices": validate_indices6,
        "name": name6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    params7 = np.array([1, 2, 3])
    indices7 = np.array([0], dtype=np.int64)
    validate_indices7 = True
    name7 = "gather_test_7"

    input_dict7 = {
        "params": params7,
        "indices": indices7,
        "validate_indices": validate_indices7,
        "name": name7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    params8 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    indices8 = np.array([1, 1, 1], dtype=np.int32)
    validate_indices8 = False
    name8 = "gather_test_8"

    input_dict8 = {
        "params": params8,
        "indices": indices8,
        "validate_indices": validate_indices8,
        "name": name8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    return list_of_inputs

generated_inputs["tf.raw_ops.Gather"] = tf_raw_ops_gather_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Gather' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Gather'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Gather', generated_inputs['tf.raw_ops.Gather'], lib="tf", suffix=0)
