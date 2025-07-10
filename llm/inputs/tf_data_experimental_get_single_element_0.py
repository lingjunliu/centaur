
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np

def tf_data_experimental_get_single_element_inputs():
    list_of_inputs = []

    # Input 1: Dataset with a single integer element
    element = np.array(1)
    dataset = tf.data.Dataset.from_tensors(element)
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 2: Dataset with a single float element
    element = np.array(3.14)
    dataset = tf.data.Dataset.from_tensors(element)
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 3: Dataset with a single string element
    element = np.array("hello")
    dataset = tf.data.Dataset.from_tensors(element)
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 4: Dataset with a single numpy array element (1D)
    element = np.array([1, 2, 3])
    dataset = tf.data.Dataset.from_tensors(element)
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 5: Dataset with a single numpy array element (2D)
    element = np.array([[1, 2], [3, 4]])
    dataset = tf.data.Dataset.from_tensors(element)
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 6: Dataset with a single numpy array element (3D)
    element = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    dataset = tf.data.Dataset.from_tensors(element)
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 7: Dataset with a single element of mixed data types (tuple)
    element = (np.array(1), np.array("hello"), np.array([1.0, 2.0]))
    dataset = tf.data.Dataset.from_tensors(element)
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 8: Dataset with a single element of mixed data types (dictionary)
    element = {"a": np.array(1), "b": np.array("hello")}
    dataset = tf.data.Dataset.from_tensors(element)
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 9: Dataset with a single negative integer element
    element = np.array(-5)
    dataset = tf.data.Dataset.from_tensors(element)
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)
    
    # Input 10: Dataset with a single element of complex type
    element = np.array([1+1j, 2+2j])
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
