
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_edit_distance_inputs():
    list_of_inputs = []

    # Input 1
    hypothesis_indices = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int64)
    hypothesis_values = np.array([b"a", b"b"], dtype=np.object_)
    hypothesis_shape = np.array([2, 1, 1], dtype=np.int64)
    hypothesis = tf.SparseTensor(hypothesis_indices, hypothesis_values, hypothesis_shape)

    truth_indices = np.array([[0, 1, 0], [1, 0, 0], [1, 0, 1], [1, 1, 0]], dtype=np.int64)
    truth_values = np.array([b"a", b"b", b"c", b"a"], dtype=np.object_)
    truth_shape = np.array([2, 2, 2], dtype=np.int64)
    truth = tf.SparseTensor(truth_indices, truth_values, truth_shape)

    normalize = True
    name = "edit_distance_1"

    input_dict = {
        "hypothesis": hypothesis,
        "truth": truth,
        "normalize": normalize,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    hypothesis_indices = np.array([[0, 0, 0], [0, 0, 1]], dtype=np.int64)
    hypothesis_values = np.array([b"a", b"b"], dtype=np.object_)
    hypothesis_shape = np.array([1, 1, 2], dtype=np.int64)
    hypothesis = tf.SparseTensor(hypothesis_indices, hypothesis_values, hypothesis_shape)

    truth_indices = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]], dtype=np.int64)
    truth_values = np.array([b"a", b"b", b"c"], dtype=np.object_)
    truth_shape = np.array([1, 1, 3], dtype=np.int64)
    truth = tf.SparseTensor(truth_indices, truth_values, truth_shape)

    normalize = False
    name = "edit_distance_2"

    input_dict = {
        "hypothesis": hypothesis,
        "truth": truth,
        "normalize": normalize,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    hypothesis_indices = np.array([[0, 0, 0]], dtype=np.int64)
    hypothesis_values = np.array([b"a"], dtype=np.object_)
    hypothesis_shape = np.array([1, 1, 1], dtype=np.int64)
    hypothesis = tf.SparseTensor(hypothesis_indices, hypothesis_values, hypothesis_shape)

    truth_indices = np.array([[0, 0, 0], [0, 0, 1]], dtype=np.int64)
    truth_values = np.array([b"a", b"b"], dtype=np.object_)
    truth_shape = np.array([1, 1, 2], dtype=np.int64)
    truth = tf.SparseTensor(truth_indices, truth_values, truth_shape)

    normalize = True
    name = "edit_distance_3"

    input_dict = {
        "hypothesis": hypothesis,
        "truth": truth,
        "normalize": normalize,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    hypothesis_indices = np.array([[0, 0, 0], [0, 1, 0]], dtype=np.int64)
    hypothesis_values = np.array([b"a", b"b"], dtype=np.object_)
    hypothesis_shape = np.array([1, 2, 1], dtype=np.int64)
    hypothesis = tf.SparseTensor(hypothesis_indices, hypothesis_values, hypothesis_shape)

    truth_indices = np.array([[0, 0, 0], [0, 1, 0]], dtype=np.int64)
    truth_values = np.array([b"c", b"d"], dtype=np.object_)
    truth_shape = np.array([1, 2, 1], dtype=np.int64)
    truth = tf.SparseTensor(truth_indices, truth_values, truth_shape)

    normalize = False
    name = "edit_distance_4"

    input_dict = {
        "hypothesis": hypothesis,
        "truth": truth,
        "normalize": normalize,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    hypothesis_indices = np.array([[0, 0, 0]], dtype=np.int64)
    hypothesis_values = np.array([b"a"], dtype=np.object_)
    hypothesis_shape = np.array([1, 1, 1], dtype=np.int64)
    hypothesis = tf.SparseTensor(hypothesis_indices, hypothesis_values, hypothesis_shape)

    truth_indices = np.array([[0, 0, 0]], dtype=np.int64)
    truth_values = np.array([b"a"], dtype=np.object_)
    truth_shape = np.array([1, 1, 1], dtype=np.int64)
    truth = tf.SparseTensor(truth_indices, truth_values, truth_shape)

    normalize = True
    name = "edit_distance_5"

    input_dict = {
        "hypothesis": hypothesis,
        "truth": truth,
        "normalize": normalize,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.edit_distance"] = tf_edit_distance_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.edit_distance' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.edit_distance'.")

check_valid('tf.edit_distance', generated_inputs['tf.edit_distance'], lib="tf", suffix=0)
