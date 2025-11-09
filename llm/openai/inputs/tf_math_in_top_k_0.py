
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_in_top_k_inputs():
    list_of_inputs = []

    # Input 1: Example-like
    targets = np.array([0, 1, 3], dtype=np.int32)
    predictions = np.array([
        [1.2, -0.3, 2.8, 5.2],
        [0.1, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.3, 0.3]
    ], dtype=np.float32)
    k = 2
    name = "case_doc_like"
    list_of_inputs.append(copy.deepcopy({
        "targets": targets, "predictions": predictions, "k": k, "name": name
    }))

    # Input 2: Ties across boundary
    targets = np.array([2, 4], dtype=np.int64)
    predictions = np.array([
        [1.0, 1.0, 1.0, 0.0, 0.0],
        [-1.0, -1.0, -1.0, -1.0, -1.0]
    ], dtype=np.float32)
    k = 2
    name = "ties_across_boundary"
    list_of_inputs.append(copy.deepcopy({
        "targets": targets, "predictions": predictions, "k": k, "name": name
    }))

    # Input 3: k equals number of classes
    targets = np.array([1, 0], dtype=np.int32)
    predictions = np.array([
        [0.0, 0.0, -1.0],
        [2.0, -2.0, 0.001]
    ], dtype=np.float32)
    k = 3
    name = "k_equals_classes"
    list_of_inputs.append(copy.deepcopy({
        "targets": targets, "predictions": predictions, "k": k, "name": name
    }))

    # Input 4: Single-class per example with non-finite values
    targets = np.array([0, 0, 0, 0], dtype=np.int64)
    predictions = np.array([
        [-0.1],
        [0.0],
        [np.nan],
        [np.inf]
    ], dtype=np.float32)
    k = 1
    name = "single_class_non_finite"
    list_of_inputs.append(copy.deepcopy({
        "targets": targets, "predictions": predictions, "k": k, "name": name
    }))

    # Input 5: Negative values and -inf at target class
    targets = np.array([2, 1, 3], dtype=np.int32)
    predictions = np.array([
        [-5.0, -0.2, -0.1, -0.3],
        [0.3, -np.inf, 0.3, 0.3],
        [10.0, 9.0, 8.0, 7.0]
    ], dtype=np.float32)
    k = 2
    name = "negatives_and_inf_target"
    list_of_inputs.append(copy.deepcopy({
        "targets": targets, "predictions": predictions, "k": k, "name": name
    }))

    # Input 6: Zero batch size
    targets = np.array([], dtype=np.int32)
    predictions = np.empty((0, 3), dtype=np.float32)
    k = 1
    name = "zero_batch"
    list_of_inputs.append(copy.deepcopy({
        "targets": targets, "predictions": predictions, "k": k, "name": name
    }))

    # Input 7: Larger class count with linspace patterns
    targets = np.array([0, 49], dtype=np.int64)
    row1 = np.linspace(-1.0, 1.0, 50, dtype=np.float32)
    row2 = np.linspace(1.0, -1.0, 50, dtype=np.float32)
    predictions = np.stack([row1, row2], axis=0).astype(np.float32)
    k = 5
    name = "large_class_count"
    list_of_inputs.append(copy.deepcopy({
        "targets": targets, "predictions": predictions, "k": k, "name": name
    }))

    # Input 8: Extreme float32 values
    targets = np.array([0, 1], dtype=np.int32)
    predictions = np.array([
        [1e38, -1e38, 0.0],
        [-1e-30, 1e-30, 0.0]
    ], dtype=np.float32)
    k = 1
    name = "extreme_float32"
    list_of_inputs.append(copy.deepcopy({
        "targets": targets, "predictions": predictions, "k": k, "name": name
    }))

    # Input 9: NaNs in non-target columns
    targets = np.array([1, 2], dtype=np.int64)
    predictions = np.array([
        [np.nan, 0.5, 0.5, 0.5],
        [0.1, np.nan, 0.2, 0.3]
    ], dtype=np.float32)
    k = 2
    name = "nans_in_non_target"
    list_of_inputs.append(copy.deepcopy({
        "targets": targets, "predictions": predictions, "k": k, "name": name
    }))

    # Input 10: All zeros with k=1 (ties)
    targets = np.array([0, 1, 2], dtype=np.int32)
    predictions = np.zeros((3, 4), dtype=np.float32)
    k = 1
    name = "all_zeros_k1"
    list_of_inputs.append(copy.deepcopy({
        "targets": targets, "predictions": predictions, "k": k, "name": name
    }))

    # Input 11: Mixed positives/negatives with k=3
    targets = np.array([4, 0, 2], dtype=np.int64)
    predictions = np.array([
        [-3.0, 0.2, 0.1, -0.5, 0.7, -0.1],
        [1.5, 0.3, 0.2, 0.1, -0.4, -0.2],
        [0.0, -0.1, 0.9, 0.8, 0.7, 0.6]
    ], dtype=np.float32)
    k = 3
    name = "mixed_values_k3"
    list_of_inputs.append(copy.deepcopy({
        "targets": targets, "predictions": predictions, "k": k, "name": name
    }))

    # Input 12: k equals classes with varied patterns
    targets = np.array([5, 3, 1], dtype=np.int32)
    predictions = np.array([
        [0.0, -1.0, -2.0, -3.0, -4.0, -5.0],
        [5.0, 4.0, -1.0, -2.0, -3.0, -4.0],
        [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ], dtype=np.float32)
    k = 6
    name = "k_equals_classes_varied"
    list_of_inputs.append(copy.deepcopy({
        "targets": targets, "predictions": predictions, "k": k, "name": name
    }))

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
