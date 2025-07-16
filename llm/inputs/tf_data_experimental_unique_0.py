
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_unique_inputs():
    list_of_inputs = []

    # Input 1
    dataset = tf.data.Dataset.from_tensor_slices(np.array([1, 37, 2, 37, 2, 1]))
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 1.0, 3.0, 2.0]))
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    dataset = tf.data.Dataset.from_tensor_slices(np.array(['a', 'b', 'a', 'c', 'b']))
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    dataset = tf.data.Dataset.from_tensor_slices(np.array([[1, 2], [3, 4], [1, 2]]))
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    dataset = tf.data.Dataset.from_tensor_slices(np.array([True, False, True, True, False]))
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    dataset = tf.data.Dataset.from_tensor_slices(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    dataset = tf.data.Dataset.from_tensor_slices(np.array([-1, -2, -1, 0, 1, 2]))
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    dataset = tf.data.Dataset.from_tensor_slices(np.array([0, 0, 0, 0, 0]))
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    dataset = tf.data.Dataset.from_tensor_slices(np.array([1, 1, 2, 2, 3, 3, 4, 4]))
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    def apply_unique(dataset):
      return dataset.apply(tf.data.experimental.unique())

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.unique"] = tf_data_experimental_unique_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.data.experimental.unique' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.unique'.")

check_valid('tf.data.experimental.unique', generated_inputs['tf.data.experimental.unique'], lib="tf", suffix=0)
