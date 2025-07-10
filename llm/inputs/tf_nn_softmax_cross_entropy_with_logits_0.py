
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_softmax_cross_entropy_with_logits_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    labels = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    logits = np.array([[2.0, 5.0, 1.0], [6.0, 1.0, 3.0]], dtype=np.float32)
    axis = -1
    name = "basic_input"
    input_dict = {"labels": labels, "logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different axis
    labels = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    logits = np.array([[2.0, 5.0, 1.0], [6.0, 1.0, 3.0]], dtype=np.float32)
    axis = 1
    name = "different_axis"
    input_dict = {"labels": labels, "logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D input
    labels = np.array([[[0.0, 1.0], [1.0, 0.0]], [[0.5, 0.5], [0.2, 0.8]]], dtype=np.float32)
    logits = np.array([[[2.0, 5.0], [6.0, 1.0]], [[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    axis = -1
    name = "3d_input"
    input_dict = {"labels": labels, "logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different dtype (float64)
    labels = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    logits = np.array([[2.0, 5.0, 1.0], [6.0, 1.0, 3.0]], dtype=np.float64)
    axis = -1
    name = "float64_input"
    input_dict = {"labels": labels, "logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Batch size of 1
    labels = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    logits = np.array([[2.0, 5.0, 1.0]], dtype=np.float32)
    axis = -1
    name = "batch_size_1"
    input_dict = {"labels": labels, "logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger number of classes
    labels = np.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]], dtype=np.float32)
    logits = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]], dtype=np.float32)
    axis = -1
    name = "larger_classes"
    input_dict = {"labels": labels, "logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Zero labels
    labels = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    logits = np.array([[2.0, 5.0, 1.0], [6.0, 1.0, 3.0]], dtype=np.float32)
    axis = -1
    name = "zero_labels"
    input_dict = {"labels": labels, "logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative Logits
    labels = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    logits = np.array([[-2.0, -5.0, -1.0], [-6.0, -1.0, -3.0]], dtype=np.float32)
    axis = -1
    name = "negative_logits"
    input_dict = {"labels": labels, "logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: All same logits
    labels = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    logits = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32)
    axis = -1
    name = "same_logits"
    input_dict = {"labels": labels, "logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: float16 data type
    labels = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float16)
    logits = np.array([[2.0, 5.0, 1.0], [6.0, 1.0, 3.0]], dtype=np.float16)
    axis = -1
    name = "float16_input"
    input_dict = {"labels": labels, "logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.softmax_cross_entropy_with_logits"] = tf_nn_softmax_cross_entropy_with_logits_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.softmax_cross_entropy_with_logits' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.softmax_cross_entropy_with_logits'.")

check_valid('tf.nn.softmax_cross_entropy_with_logits', generated_inputs['tf.nn.softmax_cross_entropy_with_logits'], lib="tf", suffix=0)
