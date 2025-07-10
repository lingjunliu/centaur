
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_confusion_matrix_inputs():
    list_of_inputs = []

    # Input 1: Basic case
    labels = np.array([1, 2, 4])
    predictions = np.array([2, 2, 4])
    num_classes = 5
    weights = None
    dtype = tf.int32
    name = "confusion_matrix_1"

    input_dict = {
        "labels": labels,
        "predictions": predictions,
        "num_classes": num_classes,
        "weights": weights,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different data type
    labels = np.array([0, 1, 2])
    predictions = np.array([0, 2, 1])
    num_classes = 3
    weights = None
    dtype = tf.int64
    name = "confusion_matrix_2"

    input_dict = {
        "labels": labels,
        "predictions": predictions,
        "num_classes": num_classes,
        "weights": weights,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Weights provided
    labels = np.array([0, 1, 2, 0, 1])
    predictions = np.array([0, 1, 1, 0, 2])
    num_classes = 3
    weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    dtype = tf.int32
    name = "confusion_matrix_3"
    input_dict = {
        "labels": labels,
        "predictions": predictions,
        "num_classes": num_classes,
        "weights": weights,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: No num_classes
    labels = np.array([0, 1, 2, 0, 1])
    predictions = np.array([0, 1, 1, 0, 2])
    num_classes = None
    weights = None
    dtype = tf.int32
    name = "confusion_matrix_4"
    input_dict = {
        "labels": labels,
        "predictions": predictions,
        "num_classes": num_classes,
        "weights": weights,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger number of classes
    labels = np.array([5, 6, 7, 8, 9])
    predictions = np.array([5, 6, 6, 8, 8])
    num_classes = 10
    weights = None
    dtype = tf.int32
    name = "confusion_matrix_5"
    input_dict = {
        "labels": labels,
        "predictions": predictions,
        "num_classes": num_classes,
        "weights": weights,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6: Zero values
    labels = np.array([0, 0, 0])
    predictions = np.array([0, 0, 0])
    num_classes = 1
    weights = None
    dtype = tf.int32
    name = "confusion_matrix_6"
    input_dict = {
        "labels": labels,
        "predictions": predictions,
        "num_classes": num_classes,
        "weights": weights,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different weights
    labels = np.array([0, 1, 2, 0, 1])
    predictions = np.array([0, 1, 1, 0, 2])
    num_classes = 3
    weights = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    dtype = tf.int32
    name = "confusion_matrix_7"
    input_dict = {
        "labels": labels,
        "predictions": predictions,
        "num_classes": num_classes,
        "weights": weights,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Single class
    labels = np.array([0, 0, 0, 0])
    predictions = np.array([0, 0, 0, 0])
    num_classes = 1
    weights = None
    dtype = tf.int32
    name = "confusion_matrix_8"

    input_dict = {
        "labels": labels,
        "predictions": predictions,
        "num_classes": num_classes,
        "weights": weights,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Unordered labels and predictions
    labels = np.array([4, 1, 0, 3, 2])
    predictions = np.array([4, 1, 0, 2, 3])
    num_classes = 5
    weights = None
    dtype = tf.int32
    name = "confusion_matrix_9"

    input_dict = {
        "labels": labels,
        "predictions": predictions,
        "num_classes": num_classes,
        "weights": weights,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: num_classes larger than max(labels, predictions)
    labels = np.array([1, 2, 4])
    predictions = np.array([2, 2, 4])
    num_classes = 10
    weights = None
    dtype = tf.int32
    name = "confusion_matrix_10"

    input_dict = {
        "labels": labels,
        "predictions": predictions,
        "num_classes": num_classes,
        "weights": weights,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.confusion_matrix"] = tf_math_confusion_matrix_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.confusion_matrix' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.confusion_matrix'.")

check_valid('tf.math.confusion_matrix', generated_inputs['tf.math.confusion_matrix'], lib="tf", suffix=0)
