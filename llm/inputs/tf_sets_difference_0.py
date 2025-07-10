
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_sets_difference_inputs():
    list_of_inputs = []

    # Input 1
    a = tf.constant([[1, 2, 3], [4, 5, 6]])
    b = tf.constant([[2, 4, 6], [5, 7, 9]])
    aminusb = True
    validate_indices = True
    input_dict = {"a": a, "b": b, "aminusb": aminusb, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    a = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    b = tf.constant([[[2, 3], [4, 5]], [[6, 7], [8, 9]]])
    aminusb = False
    validate_indices = False
    input_dict = {"a": a, "b": b, "aminusb": aminusb, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    a = tf.constant([[1, 2], [3, 4]])
    b = tf.constant([[2, 3], [4, 5]])
    aminusb = True
    validate_indices = False
    input_dict = {"a": a, "b": b, "aminusb": aminusb, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    a = tf.constant([[[1, 2, 3]]])
    b = tf.constant([[[2, 3, 4]]])
    aminusb = False
    validate_indices = True
    input_dict = {"a": a, "b": b, "aminusb": aminusb, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    a = tf.constant([[1, 5, 3, 7], [4, 5, 6, 8]])
    b = tf.constant([[2, 4, 6, 8], [5, 7, 9, 1]])
    aminusb = True
    validate_indices = True
    input_dict = {"a": a, "b": b, "aminusb": aminusb, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    a = tf.constant([1, 2, 3])
    b = tf.constant([2, 3, 4])
    aminusb = True
    validate_indices = False
    a = tf.reshape(a, (1, -1))
    b = tf.reshape(b, (1, -1))
    input_dict = {"a": a, "b": b, "aminusb": aminusb, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    a = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int64)
    b = tf.constant([[2, 4, 6], [5, 7, 9]], dtype=tf.int64)
    aminusb = False
    validate_indices = True
    input_dict = {"a": a, "b": b, "aminusb": aminusb, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8
    a = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=tf.int32)
    b = tf.constant([[[2, 3], [4, 5]], [[6, 7], [8, 9]]], dtype=tf.int32)
    aminusb = True
    validate_indices = False
    input_dict = {"a": a, "b": b, "aminusb": aminusb, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    a = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    b = tf.constant([[[2, 3, 4], [5, 6, 7]], [[8, 9, 10], [11, 12, 13]]])
    aminusb = False
    validate_indices = True
    input_dict = {"a": a, "b": b, "aminusb": aminusb, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    a = tf.constant([1, 2, 3, 4, 5])
    b = tf.constant([3, 4, 5, 6, 7])
    aminusb = True
    validate_indices = False
    a = tf.reshape(a, (1, -1))
    b = tf.reshape(b, (1, -1))
    input_dict = {"a": a, "b": b, "aminusb": aminusb, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    for input_dict in list_of_inputs:
        for k, v in input_dict.items():
            if isinstance(v, tf.Tensor):
                input_dict[k] = v.numpy()

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sets.difference"] = tf_sets_difference_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.sets.difference' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sets.difference'.")

check_valid('tf.sets.difference', generated_inputs['tf.sets.difference'], lib="tf", suffix=0)
