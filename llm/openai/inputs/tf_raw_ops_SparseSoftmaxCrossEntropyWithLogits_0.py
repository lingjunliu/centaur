
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SparseSoftmaxCrossEntropyWithLogits_inputs():
    list_of_inputs = []

    # Input 1: float32, moderate values
    features = np.array([[1.0, -1.0, 0.5, 2.0, -0.3],
                         [0.0, 0.0, 0.0, 0.0, 0.0],
                         [-2.0, 3.0, 0.1, -0.5, 1.5]], dtype=np.float32)
    labels = np.array([3, 0, 1], dtype=np.int32)
    input_dict = {"name": "case_float32_basic", "features": features, "labels": labels}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, single sample
    features = np.array([[10.0, -5.0, 0.0, 2.5]], dtype=np.float64)
    labels = np.array([2], dtype=np.int64)
    input_dict = {"name": "case_float64_single_sample", "features": features, "labels": labels}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float16, small matrix
    features = np.array([[0.5, -0.5],
                         [-1.2, 1.2]], dtype=np.float16)
    labels = np.array([0, 1], dtype=np.int32)
    input_dict = {"name": "case_float16_small", "features": features, "labels": labels}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float32, 4x3 with mix of negatives/positives
    features = np.array([[2.0, -3.0, 0.1],
                         [-1.0, 4.0, -0.2],
                         [0.0, 0.0, 0.0],
                         [5.0, -5.0, 1.0]], dtype=np.float32)
    labels = np.array([1, 1, 2, 0], dtype=np.int64)
    input_dict = {"name": "case_float32_mixed_values", "features": features, "labels": labels}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float64, wider class set
    features = np.array([[1.5, -2.5, 3.0, 0.0, -1.0, 2.2],
                         [-0.1, 0.2, -0.3, 0.4, -0.5, 0.6]], dtype=np.float64)
    labels = np.array([2, 5], dtype=np.int32)
    input_dict = {"name": "case_float64_wide_classes", "features": features, "labels": labels}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float32, single class (degenerate softmax)
    features = np.array([[0.0],
                         [1.0],
                         [-1.0],
                         [3.14],
                         [-2.71]], dtype=np.float32)
    labels = np.array([0, 0, 0, 0, 0], dtype=np.int32)
    input_dict = {"name": "case_float32_single_class", "features": features, "labels": labels}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float16, 3x4 with varied magnitudes
    features = np.array([[8.0, -8.0, 1.0, -1.0],
                         [0.25, -0.25, 0.5, -0.5],
                         [4.0, 3.0, -2.0, 1.0]], dtype=np.float16)
    labels = np.array([0, 3, 1], dtype=np.int32)
    input_dict = {"name": "case_float16_varied", "features": features, "labels": labels}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float32, extreme logits for stability
    features = np.array([[-1000.0, 0.0, 1000.0],
                         [1000.0, -1000.0, 0.0]], dtype=np.float32)
    labels = np.array([2, 0], dtype=np.int64)
    input_dict = {"name": "case_float32_extreme_logits", "features": features, "labels": labels}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float64, binary classification, larger batch
    features = np.array([[0.1, -0.1],
                         [2.0, -2.0],
                         [-3.0, 3.0],
                         [4.5, -4.5],
                         [0.0, 0.0],
                         [1.1, -1.1],
                         [-0.7, 0.7],
                         [5.0, -5.0],
                         [-2.2, 2.2],
                         [3.3, -3.3]], dtype=np.float64)
    labels = np.array([0, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int32)
    input_dict = {"name": "case_float64_binary_large_batch", "features": features, "labels": labels}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32, 2x10 multi-class
    features = np.array([[0.5, -0.2, 1.5, -1.2, 0.3, 2.0, -0.7, 0.8, -0.1, 1.0],
                         [1.2, 0.0, -0.5, 0.7, -1.0, 0.9, 0.4, -0.3, 2.5, -2.0]], dtype=np.float32)
    labels = np.array([5, 8], dtype=np.int64)
    input_dict = {"name": "case_float32_multiclass_10", "features": features, "labels": labels}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: float64, 1x1 trivial
    features = np.array([[0.0]], dtype=np.float64)
    labels = np.array([0], dtype=np.int32)
    input_dict = {"name": "case_float64_trivial_1x1", "features": features, "labels": labels}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: bfloat16 if available, else float32 fallback
    if hasattr(np, "bfloat16"):
        features = np.array([[1.0, -1.0, 0.0],
                             [0.5, 0.5, -1.0],
                             [-2.0, 1.0, 3.0]], dtype=np.bfloat16)
        labels = np.array([0, 2, 1], dtype=np.int32)
        input_dict = {"name": "case_bfloat16_available", "features": features, "labels": labels}
        list_of_inputs.append(copy.deepcopy(input_dict))
    else:
        features = np.array([[1.0, -1.0, 0.0],
                             [0.5, 0.5, -1.0],
                             [-2.0, 1.0, 3.0]], dtype=np.float32)
        labels = np.array([0, 2, 1], dtype=np.int32)
        input_dict = {"name": "case_float32_bfloat16_fallback", "features": features, "labels": labels}
        list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.SparseSoftmaxCrossEntropyWithLogits"] = tf_raw_ops_SparseSoftmaxCrossEntropyWithLogits_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SparseSoftmaxCrossEntropyWithLogits' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseSoftmaxCrossEntropyWithLogits'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.SparseSoftmaxCrossEntropyWithLogits', generated_inputs['tf.raw_ops.SparseSoftmaxCrossEntropyWithLogits'], lib="tf", suffix=0)
