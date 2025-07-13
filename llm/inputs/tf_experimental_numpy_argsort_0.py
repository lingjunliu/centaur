
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_argsort_inputs():
    list_of_inputs = []

    # Input 1: 1D array, default axis, quicksort
    a = np.array([3, 1, 4, 1, 5, 9, 2, 6])
    input_dict = {"a": a, "axis": int(-1), "kind": "quicksort", "order": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, axis=0, mergesort
    a = np.array([[3, 1, 4], [1, 5, 9], [2, 6, 5]])
    input_dict = {"a": a, "axis": int(0), "kind": "mergesort", "order": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, axis=1, heapsort
    a = np.array([[3, 1, 4], [1, 5, 9], [2, 6, 5]])
    input_dict = {"a": a, "axis": int(1), "kind": "heapsort", "order": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, axis=0
    a = np.random.rand(2, 3, 4)
    input_dict = {"a": a, "axis": int(0), "kind": "quicksort", "order": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, axis=1
    a = np.random.rand(2, 3, 4)
    input_dict = {"a": a, "axis": int(1), "kind": "quicksort", "order": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array, axis=2
    a = np.random.rand(2, 3, 4)
    input_dict = {"a": a, "axis": int(2), "kind": "quicksort", "order": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Negative values, axis=0
    a = np.array([[-3, 1, -4], [1, -5, 9], [-2, 6, -5]])
    input_dict = {"a": a, "axis": int(0), "kind": "quicksort", "order": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative values, axis=1
    a = np.array([[-3, 1, -4], [1, -5, 9], [-2, 6, -5]])
    input_dict = {"a": a, "axis": int(1), "kind": "quicksort", "order": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large array, axis=-1
    a = np.random.rand(100, 100)
    input_dict = {"a": a, "axis": int(-1), "kind": "quicksort", "order": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Array with duplicate values
    a = np.array([[1, 2, 1], [3, 1, 3]])
    input_dict = {"a": a, "axis": int(1), "kind": "quicksort", "order": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.argsort"] = tf_experimental_numpy_argsort_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.argsort' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.argsort'.")

check_valid('tf.experimental.numpy.argsort', generated_inputs['tf.experimental.numpy.argsort'], lib="tf", suffix=0)
