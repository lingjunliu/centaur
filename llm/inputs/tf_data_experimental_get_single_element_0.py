
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_get_single_element_inputs():
    list_of_inputs = []

    # Input 1: Dataset with a single numpy array
    dataset = tf.data.Dataset.from_tensors(np.array([1, 2, 3]))
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 2: Dataset with a single numpy array (2D)
    dataset = tf.data.Dataset.from_tensors(np.array([[1, 2], [3, 4]]))
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 3: Dataset with a single numpy array (string)
    dataset = tf.data.Dataset.from_tensors(np.array(["hello", "world"]))
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 4: Dataset with a single numpy array (float)
    dataset = tf.data.Dataset.from_tensors(np.array([1.0, 2.5, 3.7]))
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 5: Dataset with a single numpy array (negative values)
    dataset = tf.data.Dataset.from_tensors(np.array([-1, -2, -3]))
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 6: Dataset with a single numpy array (mixed positive and negative)
    dataset = tf.data.Dataset.from_tensors(np.array([-1, 2, -3]))
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 7: Dataset with a single numpy array (boolean)
    dataset = tf.data.Dataset.from_tensors(np.array([True, False, True]))
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 8: Dataset with a single scalar value
    dataset = tf.data.Dataset.from_tensors(np.array(5))
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
