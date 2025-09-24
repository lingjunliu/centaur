
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_sparse_softmax_cross_entropy_with_logits_inputs():
    list_of_inputs = []

    # Input 1: float32, int32
    features = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    labels = np.array([0, 2], dtype=np.int32)
    input_dict = {"features": features, "labels": labels, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, int64
    features = np.array([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]], dtype=np.float64)
    labels = np.array([2, 0], dtype=np.int64)
    input_dict = {"features": features, "labels": labels, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: bfloat16, int32
    features = np.array([[1.0, 0.0, -1.0], [-2.0, 1.0, 2.0]], dtype=np.float32).astype(np.float16)
    labels = np.array([1, 1], dtype=np.int32)
    input_dict = {"features": features, "labels": labels, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: half, int64
    features = np.array([[0.5, 0.2, 0.3], [0.9, 0.05, 0.05]], dtype=np.float32).astype(np.float16)
    labels = np.array([0, 2], dtype=np.int64)
    input_dict = {"features": features, "labels": labels, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: different batch size
    features = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    labels = np.array([3], dtype=np.int32)
    input_dict = {"features": features, "labels": labels, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: different number of classes
    features = np.array([[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]], dtype=np.float32)
    labels = np.array([0, 1, 0], dtype=np.int32)
    input_dict = {"features": features, "labels": labels, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: with name
    features = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    labels = np.array([0, 2], dtype=np.int32)
    input_dict = {"features": features, "labels": labels, "name": "my_op"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: all same label
    features = np.array([[0.1, 0.9], [0.2, 0.8]], dtype=np.float32)
    labels = np.array([1, 1], dtype=np.int32)
    input_dict = {"features": features, "labels": labels, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large values
    features = np.array([[100.0, 0.0], [0.0, 100.0]], dtype=np.float32)
    labels = np.array([0, 1], dtype=np.int32)
    input_dict = {"features": features, "labels": labels, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Zero values
    features = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    labels = np.array([0, 1], dtype=np.int32)
    input_dict = {"features": features, "labels": labels, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SparseSoftmaxCrossEntropyWithLogits"] = tf_raw_ops_sparse_softmax_cross_entropy_with_logits_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SparseSoftmaxCrossEntropyWithLogits' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseSoftmaxCrossEntropyWithLogits'.")

check_valid('tf.raw_ops.SparseSoftmaxCrossEntropyWithLogits', generated_inputs['tf.raw_ops.SparseSoftmaxCrossEntropyWithLogits'], lib="tf", suffix=0)
