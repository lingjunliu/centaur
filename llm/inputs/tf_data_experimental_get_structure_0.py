
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_get_structure_inputs():
    list_of_inputs = []

    # Input 1: Dataset from tensor slices
    dataset = tf.data.Dataset.from_tensor_slices(np.array([1, 2, 3]))
    input_dict = {"dataset_or_iterator": (dataset,)}
    list_of_inputs.append(input_dict)

    # Input 2: Dataset from a numpy array with shape (2, 3)
    dataset = tf.data.Dataset.from_tensor_slices(np.array([[1, 2, 3], [4, 5, 6]]))
    input_dict = {"dataset_or_iterator": (dataset,)}
    list_of_inputs.append(input_dict)

    # Input 3: Dataset from a numpy array with dtype float32
    dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    input_dict = {"dataset_or_iterator": (dataset,)}
    list_of_inputs.append(input_dict)

    # Input 4: Dataset from a numpy array with dtype string
    dataset = tf.data.Dataset.from_tensor_slices(np.array(['a', 'b', 'c']))
    input_dict = {"dataset_or_iterator": (dataset,)}
    list_of_inputs.append(input_dict)

    # Input 5: Dataset from a tuple of numpy arrays
    dataset = tf.data.Dataset.from_tensor_slices((np.array([1, 2, 3]), np.array(['a', 'b', 'c'])))
    input_dict = {"dataset_or_iterator": (dataset,)}
    list_of_inputs.append(input_dict)

    # Input 6: Iterator from tensor slices
    dataset = tf.data.Dataset.from_tensor_slices(np.array([1, 2, 3]))
    iterator = iter(dataset)
    input_dict = {"dataset_or_iterator": (iterator,)}
    list_of_inputs.append(input_dict)

    # Input 7: Dataset with nested structure (tuple of datasets)
    dataset1 = tf.data.Dataset.from_tensor_slices(np.array([1, 2, 3]))
    dataset2 = tf.data.Dataset.from_tensor_slices(np.array(['a', 'b', 'c']))
    dataset = tf.data.Dataset.zip((dataset1, dataset2))
    input_dict = {"dataset_or_iterator": (dataset,)}
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.get_structure"] = tf_data_experimental_get_structure_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.data.experimental.get_structure' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.get_structure'.")

check_valid('tf.data.experimental.get_structure', generated_inputs['tf.data.experimental.get_structure'], lib="tf", suffix=0)
