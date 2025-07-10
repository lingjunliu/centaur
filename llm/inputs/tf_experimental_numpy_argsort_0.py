
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
    input_dict = {"a": tf.convert_to_tensor(a), "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    a = np.array([[0, 3], [2, 2]])
    axis = 0
    kind = 'mergesort'
    order = None
    input_dict = {"a": tf.convert_to_tensor(a), "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    a = np.array([[0, 3], [2, 2]])
    axis = 1
    kind = 'heapsort'
    order = None
    input_dict = {"a": tf.convert_to_tensor(a), "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    a = np.array([1, 4, 2, 3])
    axis = 0
    kind = 'stable'
    order = None
    input_dict = {"a": tf.convert_to_tensor(a), "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    a = np.array([1, 4, 2, 3]).reshape((2,2))
    axis = -1
    kind = 'quicksort'
    order = None
    input_dict = {"a": tf.convert_to_tensor(a), "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    a = np.array([1, 4, 2, 3]).reshape((2,2))
    axis = 0
    kind = 'quicksort'
    order = None
    input_dict = {"a": tf.convert_to_tensor(a), "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7
    a = np.array([1, 4, 2, 3]).reshape((2,2))
    axis = 1
    kind = 'quicksort'
    order = None
    input_dict = {"a": tf.convert_to_tensor(a), "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    a = np.array([3, 1, 4, 1, 5, 9, 2, 6]).reshape((2,2,2))
    axis = 0
    kind = 'quicksort'
    order = None
    input_dict = {"a": tf.convert_to_tensor(a), "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    a = np.array([3, 1, 4, 1, 5, 9, 2, 6]).reshape((2,2,2))
    axis = 1
    kind = 'quicksort'
    order = None
    input_dict = {"a": tf.convert_to_tensor(a), "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    a = np.array([3, 1, 4, 1, 5, 9, 2, 6]).reshape((2,2,2))
    axis = 2
    kind = 'quicksort'
    order = None
    input_dict = {"a": tf.convert_to_tensor(a), "axis": axis, "kind": kind, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()

def convert_to_numpy(input_dict):
  new_dict = {}
  for k, v in input_dict.items():
    if isinstance(v, tf.Tensor):
      new_dict[k] = v.numpy()
    else:
      new_dict[k] = v
  return new_dict

temp_list = tf_experimental_numpy_argsort_inputs()
numpy_list = []
for item in temp_list:
  numpy_list.append(convert_to_numpy(item))

generated_inputs["tf.experimental.numpy.argsort"] = numpy_list

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.argsort' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.argsort'.")

check_valid('tf.experimental.numpy.argsort', generated_inputs['tf.experimental.numpy.argsort'], lib="tf", suffix=0)
