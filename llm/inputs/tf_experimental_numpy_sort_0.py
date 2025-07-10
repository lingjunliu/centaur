
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_sort_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D array
    a = np.array([3, 1, 4, 1, 5, 9, 2, 6])
    axis = -1
    kind = 'quicksort'
    order = None
    input_dict = {"a": a, "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, axis=0
    a = np.array([[3, 1, 4], [1, 5, 9], [2, 6, 5]])
    axis = 0
    kind = 'quicksort'
    order = None
    input_dict = {"a": a, "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, axis=1
    a = np.array([[3, 1, 4], [1, 5, 9], [2, 6, 5]])
    axis = 1
    kind = 'quicksort'
    order = None
    input_dict = {"a": a, "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, axis=0
    a = np.random.randint(0, 10, size=(2, 3, 4))
    axis = 0
    kind = 'quicksort'
    order = None
    input_dict = {"a": a, "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, axis=1
    a = np.random.randint(0, 10, size=(2, 3, 4))
    axis = 1
    kind = 'quicksort'
    order = None
    input_dict = {"a": a, "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array, axis=2
    a = np.random.randint(0, 10, size=(2, 3, 4))
    axis = 2
    kind = 'quicksort'
    order = None
    input_dict = {"a": a, "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Negative values
    a = np.array([-3, 1, -4, 1, -5, 9, -2, 6])
    axis = -1
    kind = 'quicksort'
    order = None
    input_dict = {"a": a, "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Float values
    a = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
    axis = -1
    kind = 'quicksort'
    order = None
    input_dict = {"a": a, "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: String values
    a = np.array(['b', 'a', 'c'])
    axis = -1
    kind = 'quicksort'
    order = None
    input_dict = {"a": a, "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Order
    a = np.array([('a', 2), ('c', 1), ('b', 3)], dtype=[('x', 'U1'), ('y', int)])
    axis = 0
    kind = 'quicksort'
    order = ['y', 'x']
    input_dict = {"a": a, "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.sort"] = tf_experimental_numpy_sort_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.sort' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.sort'.")

check_valid('tf.experimental.numpy.sort', generated_inputs['tf.experimental.numpy.sort'], lib="tf", suffix=0)
