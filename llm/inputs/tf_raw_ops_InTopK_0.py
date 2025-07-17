
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_in_top_k_inputs():
    list_of_inputs = []

    # Input 1
    predictions = np.array([[0.1, 0.2, 0.7], [0.9, 0.05, 0.05]], dtype=np.float32)
    targets = np.array([2, 0], dtype=np.int32)
    k = 1
    input_dict = {"predictions": predictions, "targets": targets, "k": k, "name": "test1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    predictions = np.array([[0.1, 0.2, 0.7], [0.9, 0.05, 0.05]], dtype=np.float32)
    targets = np.array([0, 1], dtype=np.int32)
    k = 2
    input_dict = {"predictions": predictions, "targets": targets, "k": k, "name": "test2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    predictions = np.array([[0.1, 0.2, 0.7], [0.9, 0.05, 0.05]], dtype=np.float32)
    targets = np.array([2, 0], dtype=np.int64)
    k = 3
    input_dict = {"predictions": predictions, "targets": targets, "k": k, "name": "test3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4 - Larger batch size
    predictions = np.random.rand(10, 5).astype(np.float32)
    targets = np.random.randint(0, 5, size=10, dtype=np.int32)
    k = 3
    input_dict = {"predictions": predictions, "targets": targets, "k": k, "name": "test4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5 - Larger number of classes
    predictions = np.random.rand(2, 100).astype(np.float32)
    targets = np.random.randint(0, 100, size=2, dtype=np.int32)
    k = 10
    input_dict = {"predictions": predictions, "targets": targets, "k": k, "name": "test5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 - k = 1
    predictions = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]], dtype=np.float32)
    targets = np.array([0, 1], dtype=np.int64)
    k = 1
    input_dict = {"predictions": predictions, "targets": targets, "k": k, "name": "test6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 - All equal predictions
    predictions = np.array([[0.33, 0.33, 0.34], [0.33, 0.33, 0.34]], dtype=np.float32)
    targets = np.array([0, 1], dtype=np.int32)
    k = 2
    input_dict = {"predictions": predictions, "targets": targets, "k": k, "name": "test7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 - Different k values
    predictions = np.array([[0.1, 0.2, 0.7], [0.9, 0.05, 0.05]], dtype=np.float32)
    targets = np.array([2, 0], dtype=np.int32)
    k = 1
    input_dict = {"predictions": predictions, "targets": targets, "k": k, "name": "test8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 - Batch Size = 1
    predictions = np.array([[0.2, 0.3, 0.5]], dtype=np.float32)
    targets = np.array([2], dtype=np.int64)
    k = 1
    input_dict = {"predictions": predictions, "targets": targets, "k": k, "name": "test9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 - k equals the number of classes
    predictions = np.array([[0.1, 0.2, 0.7], [0.9, 0.05, 0.05]], dtype=np.float32)
    targets = np.array([2, 0], dtype=np.int32)
    k = 3
    input_dict = {"predictions": predictions, "targets": targets, "k": k, "name": "test10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.InTopK"] = tf_raw_ops_in_top_k_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.InTopK' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.InTopK'.")

check_valid('tf.raw_ops.InTopK', generated_inputs['tf.raw_ops.InTopK'], lib="tf", suffix=0)
