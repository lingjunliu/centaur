
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
    labels = np.array([0, 1], dtype=np.int32)
    input_dict = {"features": features, "labels": labels, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, int64
    features = np.array([[0.1, 0.2, 0.7], [0.4, 0.5, 0.1]], dtype=np.float64)
    labels = np.array([2, 0], dtype=np.int64)
    input_dict = {"features": features, "labels": labels, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: half, int32
    features = np.array([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]], dtype=np.float16) # Use np.float16 for half type
    labels = np.array([1, 2], dtype=np.int32)
    input_dict = {"features": features, "labels": labels, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: bfloat16, int64
    features = np.array([[-1.0, 0.0, 1.0], [2.0, 1.0, 0.0]], dtype=tf.bfloat16.as_numpy_dtype)
    labels = np.array([0, 2], dtype=np.int64)
    input_dict = {"features": features, "labels": labels, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32, int32, different batch size and num_classes
    features = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 5.0, 6.0, 7.0], [7.0, 8.0, 9.0, 10.0]], dtype=np.float32)
    labels = np.array([0, 1, 3], dtype=np.int32)
    input_dict = {"features": features, "labels": labels, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float64, int64, negative values in features
    features = np.array([[-0.1, -0.2, -0.7], [-0.4, -0.5, -0.1]], dtype=np.float64)
    labels = np.array([2, 0], dtype=np.int64)
    input_dict = {"features": features, "labels": labels, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: half, int32, larger values
    features = np.array([[1000.5, 2000.5, 3000.5], [4000.5, 5000.5, 6000.5]], dtype=np.float16)
    labels = np.array([1, 2], dtype=np.int32)
    input_dict = {"features": features, "labels": labels, "name": "test_name"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: bfloat16, int64, zero values
    features = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=tf.bfloat16.as_numpy_dtype)
    labels = np.array([0, 2], dtype=np.int64)
    input_dict = {"features": features, "labels": labels, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9: float32, int32 with labels close to num_classes
    features = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    labels = np.array([2, 1], dtype=np.int32)
    input_dict = {"features": features, "labels": labels, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float64, int64, small values in features
    features = np.array([[0.001, 0.002, 0.007], [0.004, 0.005, 0.001]], dtype=np.float64)
    labels = np.array([2, 0], dtype=np.int64)
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
