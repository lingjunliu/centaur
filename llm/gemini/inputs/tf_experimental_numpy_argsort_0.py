
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_argsort_inputs():
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
    axis = 0
    kind = 'quicksort'
    order = None
    input_dict = {"a": a, "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    a = np.array([[3, 1, 4], [1, 5, 9]])
    axis = 1
    kind = 'quicksort'
    order = None
    input_dict = {"a": a, "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    a = np.array([1.1, 1.0, 1.2])
    axis = -1
    kind = 'quicksort'
    order = None
    input_dict = {"a": a, "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    a = np.array([1, 2, 3], dtype=np.int32)
    axis = -1
    kind = 'quicksort'
    order = None
    input_dict = {"a": a, "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    a = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
    axis = 0
    kind = 'quicksort'
    order = None
    input_dict = {"a": a, "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    a = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
    axis = 1
    kind = 'quicksort'
    order = None
    input_dict = {"a": a, "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    a = np.array([0.5, 0.2, 0.9, 0.1])
    axis = -1
    kind = 'quicksort'
    order = None
    input_dict = {"a": a, "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    a = np.array([5, 2, 8, 1, 9, 4, 7, 3, 6])
    axis = -1
    kind = 'quicksort'
    order = None
    input_dict = {"a": a, "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 10
    a = np.array([[-1, -2], [-3, -4]])
    axis = 0
    kind = 'quicksort'
    order = None
    input_dict = {"a": a, "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    a = np.array([[-1, -2], [-3, -4]])
    axis = 1
    kind = 'quicksort'
    order = None
    input_dict = {"a": a, "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    a = np.array([1.0, np.nan, 2.0, np.nan])
    axis = -1
    kind = 'quicksort'
    order = None
    input_dict = {"a": a, "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13
    a = np.array([True, False, True])
    axis = -1
    kind = 'quicksort'
    order = None
    input_dict = {"a": a, "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.argsort"] = tf_argsort_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.argsort' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.argsort'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.argsort', generated_inputs['tf.experimental.numpy.argsort'], lib="tf", suffix=0)
