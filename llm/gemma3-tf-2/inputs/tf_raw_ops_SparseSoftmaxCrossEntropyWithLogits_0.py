
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def sparse_softmax_cross_entropy_with_logits_inputs():
    list_of_inputs = []

    features = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    labels = np.array([0, 1], dtype=np.int32)
    input_dict = {'name': 'op1', 'features': features, 'labels': labels}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[-1.0, 0.5, 1.2], [2.3, -0.8, 0.1]], dtype=np.float64)
    labels = np.array([2, 0], dtype=np.int64)
    input_dict = {'name': 'op2', 'features': features, 'labels': labels}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=np.float32)
    labels = np.array([1, 3], dtype=np.int32)
    input_dict = {'name': 'op3', 'features': features, 'labels': labels}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]], dtype=np.float64)
    labels = np.array([1, 2], dtype=np.int32)
    input_dict = {'name': 'op5', 'features': features, 'labels': labels}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    labels = np.array([0, 0], dtype=np.int64)
    input_dict = {'name': 'op6', 'features': features, 'labels': labels}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]], dtype=np.float32)
    labels = np.array([2, 1], dtype=np.int32)
    input_dict = {'name': 'op7', 'features': features, 'labels': labels}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
    labels = np.array([3], dtype=np.int32)
    input_dict = {'name': 'op9', 'features': features, 'labels': labels}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    labels = np.array([0, 1, 0], dtype=np.int64)
    input_dict = {'name': 'op10', 'features': features, 'labels': labels}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.SparseSoftmaxCrossEntropyWithLogits"] = sparse_softmax_cross_entropy_with_logits_inputs()

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
