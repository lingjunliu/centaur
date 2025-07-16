
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_get_structure_inputs():
    list_of_inputs = []

    # Input 1: Dataset from tensor slices
    dataset = tf.data.Dataset.from_tensor_slices(np.array([1, 2, 3], dtype=np.int32))
    input_dict = {"dataset_or_iterator": (dataset,)}
    list_of_inputs.append(input_dict)

    # Input 2: Dataset from list of tuples, ensuring consistent types - USE TENSORS!
    example = [(tf.constant(1, dtype=tf.int32), tf.constant('a'.encode('utf-8'), dtype=tf.string)), (tf.constant(2, dtype=tf.int32), tf.constant('b'.encode('utf-8'), dtype=tf.string)), (tf.constant(3, dtype=tf.int32), tf.constant('c'.encode('utf-8'), dtype=tf.string))]
    dataset = tf.data.Dataset.from_tensor_slices(example)
    input_dict = {"dataset_or_iterator": (dataset,)}
    list_of_inputs.append(input_dict)

    # Input 3: Iterator from dataset
    dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    iterator = iter(dataset)
    input_dict = {"dataset_or_iterator": (iterator,)}
    list_of_inputs.append(input_dict)

    # Input 4: Dataset with multiple dimensions
    dataset = tf.data.Dataset.from_tensor_slices(np.array([[1, 2], [3, 4]], dtype=np.int64))
    input_dict = {"dataset_or_iterator": (dataset,)}
    list_of_inputs.append(input_dict)

    # Input 5: Dataset with strings
    dataset = tf.data.Dataset.from_tensor_slices(np.array(['a'.encode('utf-8'), 'b'.encode('utf-8'), 'c'.encode('utf-8')], dtype=tf.string))
    input_dict = {"dataset_or_iterator": (dataset,)}
    list_of_inputs.append(input_dict)

    # Input 6: Dataset with tuples of different types and shapes - USE TENSORS!
    example = [(tf.constant([1, 2], dtype=tf.int32), tf.constant('a'.encode('utf-8'), dtype=tf.string)), (tf.constant([3, 4], dtype=tf.int32), tf.constant('b'.encode('utf-8'), dtype=tf.string))]
    dataset = tf.data.Dataset.from_tensor_slices(example)
    input_dict = {"dataset_or_iterator": (dataset,)}
    list_of_inputs.append(input_dict)

    # Input 7: Dataset with nested structures - USE TENSORS!
    example = [((tf.constant(1, dtype=tf.int32), tf.constant(2, dtype=tf.int32)), tf.constant(3, dtype=tf.int32)), ((tf.constant(4, dtype=tf.int32), tf.constant(5, dtype=tf.int32)), tf.constant(6, dtype=tf.int32))]
    dataset = tf.data.Dataset.from_tensor_slices(example)
    input_dict = {"dataset_or_iterator": (dataset,)}
    list_of_inputs.append(input_dict)

     # Input 8: Dataset with boolean values
    dataset = tf.data.Dataset.from_tensor_slices(np.array([True, False, True], dtype=np.bool_))
    input_dict = {"dataset_or_iterator": (dataset,)}
    list_of_inputs.append(input_dict)

    # Input 9: Dataset with complex numbers
    dataset = tf.data.Dataset.from_tensor_slices(np.array([1+1j, 2+2j, 3+3j], dtype=np.complex128))
    input_dict = {"dataset_or_iterator": (dataset,)}
    list_of_inputs.append(input_dict)

    # Input 10: Iterator from dataset with different types - USE TENSORS!
    int_data = tf.constant([1, 2, 3], dtype=tf.int32)
    str_data = tf.constant(['a'.encode('utf-8'), 'b'.encode('utf-8'), 'c'.encode('utf-8')], dtype=tf.string)
    dataset = tf.data.Dataset.from_tensor_slices((int_data, str_data))

    iterator = iter(dataset)
    input_dict = {"dataset_or_iterator": (iterator,)}
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
