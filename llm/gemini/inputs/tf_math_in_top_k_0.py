
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_in_top_k_inputs():
    list_of_inputs = []

    # Input 1: Basic example
    targets = np.array([0, 1, 3], dtype=np.int32)
    predictions = np.array([
        [1.2, -0.3, 2.8, 5.2],
        [0.1, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.3, 0.3]
    ], dtype=np.float32)
    k = 2
    name = "case1"
    list_of_inputs.append({
        'targets': targets,
        'predictions': predictions,
        'k': k,
        'name': name
    })

    # Input 2: Int64 targets and smaller batch
    targets = np.array([1, 0], dtype=np.int64)
    predictions = np.array([
        [-1.0, 2.0],
        [3.0, -1.0]
    ], dtype=np.float32)
    k = 1
    name = "case2"
    list_of_inputs.append({
        'targets': targets,
        'predictions': predictions,
        'k': k,
        'name': name
    })

    # Input 3: Larger batch with varying values
    targets = np.array([2, 2, 1, 0], dtype=np.int32)
    predictions = np.array([
        [0.1, 0.2, 0.7],
        [0.1, 0.8, 0.1],
        [0.5, 0.4, 0.1],
        [0.9, 0.0, 0.1]
    ], dtype=np.float32)
    k = 2
    name = "case3"
    list_of_inputs.append({
        'targets': targets,
        'predictions': predictions,
        'k': k,
        'name': name
    })

    # Input 4: Minimum batch size (1)
    targets = np.array([0], dtype=np.int32)
    predictions = np.array([[1.5]], dtype=np.float32)
    k = 1
    name = "case4"
    list_of_inputs.append({
        'targets': targets,
        'predictions': predictions,
        'k': k,
        'name': name
    })

    # Input 5: k equals number of classes
    targets = np.array([1, 2, 0, 1, 2], dtype=np.int32)
    predictions = np.arange(15, dtype=np.float32).reshape(5, 3)
    k = 3
    name = "case5"
    list_of_inputs.append({
        'targets': targets,
        'predictions': predictions,
        'k': k,
        'name': name
    })

    # Input 6: Negative predictions
    targets = np.array([2, 0], dtype=np.int64)
    predictions = np.array([
        [-10.0, -5.0, -1.0],
        [-1.0, -2.0, -3.0]
    ], dtype=np.float32)
    k = 1
    name = "case6"
    list_of_inputs.append({
        'targets': targets,
        'predictions': predictions,
        'k': k,
        'name': name
    })

    # Input 7: Identity targets with k = 3
    targets = np.array([0, 1, 2], dtype=np.int32)
    predictions = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ], dtype=np.float32)
    k = 3
    name = "case7"
    list_of_inputs.append({
        'targets': targets,
        'predictions': predictions,
        'k': k,
        'name': name
    })

    # Input 8: One-hot predictions
    targets = np.array([4, 3, 2, 1, 0], dtype=np.int32)
    predictions = np.eye(5, dtype=np.float32)
    k = 4
    name = "case8"
    list_of_inputs.append({
        'targets': targets,
        'predictions': predictions,
        'k': k,
        'name': name
    })

    # Input 9: All identical values (ties test)
    targets = np.array([0, 1], dtype=np.int32)
    predictions = np.array([
        [5.0, 5.0, 5.0],
        [5.0, 5.0, 5.0]
    ], dtype=np.float32)
    k = 1
    name = "case9"
    list_of_inputs.append({
        'targets': targets,
        'predictions': predictions,
        'k': k,
        'name': name
    })

    # Input 10: Large values
    targets = np.array([1, 1], dtype=np.int64)
    predictions = np.array([
        [1e5, 1e6],
        [1e6, 1e5]
    ], dtype=np.float32)
    k = 1
    name = "case10"
    list_of_inputs.append({
        'targets': targets,
        'predictions': predictions,
        'k': k,
        'name': name
    })

    return list_of_inputs

generated_inputs["tf.math.in_top_k"] = tf_math_in_top_k_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.in_top_k' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.in_top_k'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.in_top_k', generated_inputs['tf.math.in_top_k'], lib="tf", suffix=0)
