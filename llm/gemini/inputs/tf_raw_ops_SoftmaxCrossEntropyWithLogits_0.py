
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SoftmaxCrossEntropyWithLogits_inputs():
    list_of_inputs = []

    # Input 1: float32, basic example
    features = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    labels = np.array([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]], dtype=np.float32)
    input_dict = {"features": features, "labels": labels, "name": "example1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, different values
    features = np.array([[-1.0, 0.0, 1.0], [2.0, -3.0, 0.5]], dtype=np.float64)
    labels = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]], dtype=np.float64)
    input_dict = {"features": features, "labels": labels, "name": "example2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: half, small values
    features = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float16)
    labels = np.array([[0.6, 0.4], [0.5, 0.5]], dtype=np.float16)
    input_dict = {"features": features, "labels": labels, "name": "example3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float32 instead of bfloat16
    features = np.array([[1.1, 1.2], [1.3, 1.4]], dtype=np.float32)
    labels = np.array([[0.4, 0.6], [0.7, 0.3]], dtype=np.float32)
    input_dict = {"features": features, "labels": labels, "name": "example4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32, single batch
    features = np.array([[1.0, 2.0]], dtype=np.float32)
    labels = np.array([[0.3, 0.7]], dtype=np.float32)
    input_dict = {"features": features, "labels": labels, "name": "example5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float64, single class
    features = np.array([[1.0], [2.0]], dtype=np.float64)
    labels = np.array([[1.0], [1.0]], dtype=np.float64)
    input_dict = {"features": features, "labels": labels, "name": "example6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32, larger batch size
    features = np.random.rand(10, 5).astype(np.float32)
    labels = np.random.rand(10, 5).astype(np.float32)
    labels = labels / np.sum(labels, axis=1, keepdims=True) # Normalize to ensure valid probability distributions
    input_dict = {"features": features, "labels": labels, "name": "example7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float64, larger number of classes
    features = np.random.rand(2, 20).astype(np.float64)
    labels = np.random.rand(2, 20).astype(np.float64)
    labels = labels / np.sum(labels, axis=1, keepdims=True)
    input_dict = {"features": features, "labels": labels, "name": "example8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: half, all negative values
    features = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float16)
    labels = np.array([[0.2, 0.8], [0.9, 0.1]], dtype=np.float16)
    input_dict = {"features": features, "labels": labels, "name": "example9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32 instead of bfloat16, mixed positive and negative
    features = np.array([[-0.5, 1.5], [2.5, -3.5]], dtype=np.float32)
    labels = np.array([[0.3, 0.7], [0.6, 0.4]], dtype=np.float32)
    input_dict = {"features": features, "labels": labels, "name": "example10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SoftmaxCrossEntropyWithLogits"] = tf_raw_ops_SoftmaxCrossEntropyWithLogits_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SoftmaxCrossEntropyWithLogits' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SoftmaxCrossEntropyWithLogits'.")

check_valid('tf.raw_ops.SoftmaxCrossEntropyWithLogits', generated_inputs['tf.raw_ops.SoftmaxCrossEntropyWithLogits'], lib="tf", suffix=0)
