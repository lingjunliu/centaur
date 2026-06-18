
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_bincount_inputs():
    list_of_inputs = []

    # Input 1: Basic case, 1D arr, size 5, empty float32 weights
    list_of_inputs.append({
        "name": "bincount_1",
        "arr": np.array([1, 2, 2, 3], dtype=np.int32),
        "size": np.array(5, dtype=np.int32),
        "weights": np.array([], dtype=np.float32)
    })

    # Input 2: 1D arr, size 4, with matching float32 weights
    list_of_inputs.append({
        "name": "bincount_2",
        "arr": np.array([0, 1, 2, 1], dtype=np.int32),
        "size": np.array(4, dtype=np.int32),
        "weights": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    })

    # Input 3: Values outside range [0, size) to be ignored
    list_of_inputs.append({
        "name": "bincount_3",
        "arr": np.array([0, 5, 2, 10], dtype=np.int32),
        "size": np.array(4, dtype=np.int32),
        "weights": np.array([], dtype=np.int32)
    })

    # Input 4: Using int64 weights
    list_of_inputs.append({
        "name": "bincount_4",
        "arr": np.array([0, 1, 2], dtype=np.int32),
        "size": np.array(3, dtype=np.int32),
        "weights": np.array([10, 20, 30], dtype=np.int64)
    })

    # Input 5: Using float64 weights
    list_of_inputs.append({
        "name": "bincount_5",
        "arr": np.array([1, 1, 1], dtype=np.int32),
        "size": np.array(2, dtype=np.int32),
        "weights": np.array([1.5, 2.5, 3.5], dtype=np.float64)
    })

    # Input 6: Empty arr and float32 empty weights
    list_of_inputs.append({
        "name": "bincount_6",
        "arr": np.array([], dtype=np.int32),
        "size": np.array(3, dtype=np.int32),
        "weights": np.array([], dtype=np.float32)
    })

    # Input 7: Values in arr outside [0, size) (must be non-negative)
    list_of_inputs.append({
        "name": "bincount_7",
        "arr": np.array([5, 10, 15], dtype=np.int32),
        "size": np.array(2, dtype=np.int32),
        "weights": np.array([], dtype=np.float32)
    })

    # Input 8: Size is 0 (all values ignored)
    list_of_inputs.append({
        "name": "bincount_8",
        "arr": np.array([1, 2, 3], dtype=np.int32),
        "size": np.array(0, dtype=np.int32),
        "weights": np.array([], dtype=np.float32)
    })

    # Input 9: Large size, empty int32 weights
    list_of_inputs.append({
        "name": "bincount_9",
        "arr": np.array([10, 20], dtype=np.int32),
        "size": np.array(30, dtype=np.int32),
        "weights": np.array([], dtype=np.int32)
    })

    # Input 10: int32 weights matching arr
    list_of_inputs.append({
        "name": "bincount_10",
        "arr": np.array([0, 2, 2, 1], dtype=np.int32),
        "size": np.array(3, dtype=np.int32),
        "weights": np.array([5, 10, 15, 20], dtype=np.int32)
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.Bincount"] = tf_raw_ops_bincount_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Bincount' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Bincount'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Bincount', generated_inputs['tf.raw_ops.Bincount'], lib="tf", suffix=0)
