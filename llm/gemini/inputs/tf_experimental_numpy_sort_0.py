
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_sort_inputs():
    list_of_inputs = []

    # Input 1
    a = np.array([3, 1, 4, 1, 5, 9, 2, 6])
    axis = -1
    kind = 'quicksort'
    order = None
    input_dict = {"a": a, "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    a = np.array([[3, 1, 4], [1, 5, 9]])
    axis = 1
    kind = 'mergesort'
    order = None
    input_dict = {"a": a, "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    a = np.array([[3, 1, 4], [1, 5, 9]])
    axis = 0
    kind = 'heapsort'
    order = None
    input_dict = {"a": a, "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    a = np.array([1.0, 2.0, 3.0, 4.0])
    axis = -1
    kind = 'quicksort'
    order = None
    input_dict = {"a": a, "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    a = np.array([[1, 2], [3, 4]])
    axis = 1
    kind = 'mergesort'
    order = None
    input_dict = {"a": a, "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    a = np.array([['c', 'a', 'b'], ['e', 'd', 'f']])
    axis = 1
    kind = 'quicksort'
    order = None
    input_dict = {"a": a, "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    a = np.array([5, 2, 8, 1, 9, 4])
    axis = -1
    kind = 'heapsort'
    order = None
    input_dict = {"a": a, "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8
    a = np.array([[9, 8, 7], [6, 5, 4]])
    axis = 0
    kind = 'quicksort'
    order = None
    input_dict = {"a": a, "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    a = np.array([[[1,3],[2,4]],[[5,7],[6,8]]])
    axis = 1
    kind = 'mergesort'
    order = None
    input_dict = {"a": a, "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    a = np.array([[[1,3],[2,4]],[[5,7],[6,8]]])
    axis = 2
    kind = 'heapsort'
    order = None
    input_dict = {"a": a, "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    a = np.array([1, 5, 2, 8])
    axis = -1
    kind = 'quicksort'
    order = None
    input_dict = {"a": a, "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.sort"] = tf_experimental_numpy_sort_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.sort' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.sort'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.sort', generated_inputs['tf.experimental.numpy.sort'], lib="tf", suffix=0)
