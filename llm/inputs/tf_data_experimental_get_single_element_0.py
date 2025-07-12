
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_get_single_element_inputs():
    list_of_inputs = []

    # Input 1: Simple dataset with one element
    element = np.array([1, 2, 3])
    dataset = tf.data.Dataset.from_tensors(element)
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 2: Dataset with one element, a tuple
    element1 = np.array([1, 2])
    element2 = np.array([3, 4])
    dataset = tf.data.Dataset.from_tensors((element1, element2))
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 3: Dataset with one element, a dictionary
    element_a = np.array([1, 2])
    element_b = np.array([3, 4])
    element = {"a": element_a, "b": element_b}
    dataset = tf.data.Dataset.from_tensors(element)
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 4: Dataset with one element, a multi-dimensional array
    element = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    dataset = tf.data.Dataset.from_tensors(element)
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 5: Dataset with one element, a string
    element = np.array("hello")
    dataset = tf.data.Dataset.from_tensors(element)
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.get_single_element"] = tf_data_experimental_get_single_element_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.data.experimental.get_single_element' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.get_single_element'.")

check_valid('tf.data.experimental.get_single_element', generated_inputs['tf.data.experimental.get_single_element'], lib="tf", suffix=0)
