
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_get_structure_inputs():
    list_of_inputs = []

    # Input 1: Dataset from a simple tensor
    dataset1 = tf.data.Dataset.from_tensor_slices(np.array([1, 2, 3], dtype=np.int32))
    input_dict1 = {"dataset_or_iterator": dataset1}
    list_of_inputs.append(input_dict1)

    # Input 2: Dataset from a tuple of tensors
    dataset2 = tf.data.Dataset.from_tensor_slices((np.array([1, 2, 3], dtype=np.int32), np.array(['a', 'b', 'c'])))
    input_dict2 = {"dataset_or_iterator": dataset2}
    list_of_inputs.append(input_dict2)

    # Input 3: Dataset from a dictionary of tensors
    dataset3 = tf.data.Dataset.from_tensor_slices({"x": np.array([1, 2, 3], dtype=np.int32), "y": np.array([4, 5, 6], dtype=np.int32)})
    input_dict3 = {"dataset_or_iterator": dataset3}
    list_of_inputs.append(input_dict3)

    # Input 4: Dataset from a nested structure
    dataset4 = tf.data.Dataset.from_tensor_slices(((np.array([1, 2], dtype=np.int32), np.array([3, 4], dtype=np.int32)), np.array([5, 6], dtype=np.int32)))
    input_dict4 = {"dataset_or_iterator": dataset4}
    list_of_inputs.append(input_dict4)

    # Input 5: Iterator from a simple dataset
    dataset5 = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    iterator5 = iter(dataset5)
    input_dict5 = {"dataset_or_iterator": iterator5}
    list_of_inputs.append(input_dict5)

    # Input 6: Iterator from a dataset of tuples
    dataset6 = tf.data.Dataset.from_tensor_slices((np.array([1, 2, 3], dtype=np.int32), np.array(['a', 'b', 'c'])))
    iterator6 = iter(dataset6)
    input_dict6 = {"dataset_or_iterator": iterator6}
    list_of_inputs.append(input_dict6)

    # Input 7: Dataset with different shapes
    dataset7 = tf.data.Dataset.from_tensor_slices(np.array([[1, 2], [3, 4]], dtype=np.int32))
    input_dict7 = {"dataset_or_iterator": dataset7}
    list_of_inputs.append(input_dict7)

    # Input 8: Dataset with tf.string type
    dataset8 = tf.data.Dataset.from_tensor_slices(np.array(["hello", "world"]))
    input_dict8 = {"dataset_or_iterator": dataset8}
    list_of_inputs.append(input_dict8)

    # Input 9: Dataset with tf.bool type
    dataset9 = tf.data.Dataset.from_tensor_slices(np.array([True, False, True]))
    input_dict9 = {"dataset_or_iterator": dataset9}
    list_of_inputs.append(input_dict9)

    # Input 10: Dataset with a combination of different dtypes
    dataset10 = tf.data.Dataset.from_tensor_slices((np.array([1, 2], dtype=np.int32), np.array([1.0, 2.0], dtype=np.float32), np.array([True, False])))
    input_dict10 = {"dataset_or_iterator": dataset10}
    list_of_inputs.append(input_dict10)

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
