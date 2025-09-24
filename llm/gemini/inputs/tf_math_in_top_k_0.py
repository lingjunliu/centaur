
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_in_top_k_inputs():
    list_of_inputs = []

    # Input 1
    targets = np.array([0, 1, 2], dtype=np.int32)
    predictions = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.2, 0.2, 0.6]], dtype=np.float32)
    k = 1
    name = "top_1"
    input_dict = {"targets": targets, "predictions": predictions, "k": k, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    targets = np.array([0, 1, 2, 0], dtype=np.int64)
    predictions = np.array([[0.1, 0.2, 0.7], [0.5, 0.4, 0.1], [0.3, 0.3, 0.4], [0.9, 0.05, 0.05]], dtype=np.float32)
    k = 2
    name = "top_2"
    input_dict = {"targets": targets, "predictions": predictions, "k": k, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    targets = np.array([0, 1], dtype=np.int32)
    predictions = np.array([[0.3, 0.7], [0.9, 0.1]], dtype=np.float32)
    k = 1
    name = None
    input_dict = {"targets": targets, "predictions": predictions, "k": k, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    targets = np.array([2, 0, 1], dtype=np.int32)
    predictions = np.array([[0.1, 0.2, 0.7], [0.9, 0.05, 0.05], [0.5, 0.4, 0.1]], dtype=np.float32)
    k = 3
    name = "top_3_all"
    input_dict = {"targets": targets, "predictions": predictions, "k": k, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    targets = np.array([0], dtype=np.int32)
    predictions = np.array([[0.8, 0.1, 0.1]], dtype=np.float32)
    k = 1
    name = "top_1_single"
    input_dict = {"targets": targets, "predictions": predictions, "k": k, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    targets = np.array([0, 1, 0, 1], dtype=np.int64)
    predictions = np.array([[0.6, 0.4], [0.3, 0.7], [0.5, 0.5], [0.1, 0.9]], dtype=np.float32)
    k = 1
    name = "ties_test"
    input_dict = {"targets": targets, "predictions": predictions, "k": k, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: k larger than number of classes
    targets = np.array([0, 1], dtype=np.int32)
    predictions = np.array([[0.6, 0.4], [0.3, 0.7]], dtype=np.float32)
    k = 3
    name = "k_greater_classes"
    input_dict = {"targets": targets, "predictions": predictions, "k": k, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative values in predictions
    targets = np.array([0, 1], dtype=np.int32)
    predictions = np.array([[-0.6, -0.4], [-0.3, -0.7]], dtype=np.float32)
    k = 1
    name = "negative_predictions"
    input_dict = {"targets": targets, "predictions": predictions, "k": k, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Same prediction scores
    targets = np.array([0, 1, 2], dtype=np.int32)
    predictions = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], dtype=np.float32)
    k = 2
    name = "same_scores"
    input_dict = {"targets": targets, "predictions": predictions, "k": k, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different batch size, different k
    targets = np.array([1, 0, 2, 1, 0], dtype=np.int64)
    predictions = np.array([[0.1, 0.9, 0.0], [0.8, 0.1, 0.1], [0.2, 0.3, 0.5], [0.2, 0.7, 0.1], [0.6, 0.3, 0.1]], dtype=np.float32)
    k = 2
    name = "diff_batch_k"
    input_dict = {"targets": targets, "predictions": predictions, "k": k, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.in_top_k"] = tf_math_in_top_k_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.in_top_k' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.in_top_k'.")

check_valid('tf.math.in_top_k', generated_inputs['tf.math.in_top_k'], lib="tf", suffix=0)
