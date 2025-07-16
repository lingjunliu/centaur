
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_argsort_inputs():
    list_of_inputs = []

    # Input 1
    a = np.array([3, 1, 4, 1, 5, 9, 2, 6])
    axis = -1
    kind = 'quicksort'
    order = None
    input_dict = {"a": a, "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    a = np.array([[0, 3], [2, 2]])
    axis = 0
    kind = 'quicksort'
    order = None
    input_dict = {"a": a, "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    a = np.array([[0, 3], [2, 2]])
    axis = 1
    kind = 'quicksort'
    order = None
    input_dict = {"a": a, "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    a = np.array([1, 2, 3], dtype=np.int32)
    axis = -1
    kind = 'quicksort'
    order = None
    input_dict = {"a": a, "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    a = np.array([5, 4, 3, 2, 1])
    axis = 0
    kind = 'mergesort'
    order = None
    input_dict = {"a": a, "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    a = np.array([[5, 4], [3, 2]])
    axis = 1
    kind = 'heapsort'
    order = None
    input_dict = {"a": a, "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    a = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    axis = 0
    kind = 'quicksort'
    order = None
    input_dict = {"a": a, "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8
    a = np.array([[9, 8, 7], [6, 5, 4]], dtype=np.int64)
    axis = 0
    kind = 'quicksort'
    order = None
    input_dict = {"a": a, "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    a = np.array([('a', 2), ('c', 1), ('b', 9)],
                 dtype=[('x', 'S1'), ('y', int)])
    axis = -1
    kind = 'quicksort'
    order = ['x', 'y']
    input_dict = {"a": a, "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    a = np.array([1, 2, 1, 4, 5], dtype=np.float32)
    axis = 0
    kind = 'quicksort'
    order = None
    input_dict = {"a": a, "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    axis = 1
    kind = 'quicksort'
    order = None
    input_dict = {"a": a, "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12
    a = np.array([1, 2, 3])
    axis = -1
    kind = 'quicksort'
    order = None
    input_dict = {"a": a, "axis": axis, "kind": kind, "order": order}
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
