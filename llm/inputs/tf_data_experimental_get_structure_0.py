
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np

def tf_data_experimental_get_structure_inputs():
    list_of_inputs = []

    # Input 1: Dataset from tensor slices
    dataset1 = tf.data.Dataset.from_tensor_slices(np.array([1, 2, 3]))
    input_dict1 = {"dataset_or_iterator": dataset1}
    list_of_inputs.append(input_dict1)

    # Input 2: Dataset from tensors
    a = tf.constant([1, 2, 3])
    b = tf.constant(['a', 'b', 'c'])
    dataset2 = tf.data.Dataset.from_tensor_slices((a, b))
    input_dict2 = {"dataset_or_iterator": dataset2}
    list_of_inputs.append(input_dict2)

    # Input 3: Dataset from tensor with different data type
    dataset3 = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0]))
    input_dict3 = {"dataset_or_iterator": dataset3}
    list_of_inputs.append(input_dict3)

    # Input 4: Dataset from tensor with different shape
    dataset4 = tf.data.Dataset.from_tensor_slices(np.array([[1, 2], [3, 4]]))
    input_dict4 = {"dataset_or_iterator": dataset4}
    list_of_inputs.append(input_dict4)

    # Input 5: Iterator from dataset
    dataset5 = tf.data.Dataset.from_tensor_slices(np.array([1, 2, 3]))
    iterator5 = iter(dataset5)
    input_dict5 = {"dataset_or_iterator": iterator5}
    list_of_inputs.append(input_dict5)

    # Input 6: Dataset from generator
    def generator():
        for i in range(3):
            yield i

    dataset6 = tf.data.Dataset.from_generator(generator, output_signature=tf.TensorSpec(shape=(), dtype=tf.int64))
    input_dict6 = {"dataset_or_iterator": dataset6}
    list_of_inputs.append(input_dict6)

    # Input 7: Dataset from numpy array of boolean values
    dataset7 = tf.data.Dataset.from_tensor_slices(np.array([True, False, True]))
    input_dict7 = {"dataset_or_iterator": dataset7}
    list_of_inputs.append(input_dict7)

    # Input 8: Dataset from strings
    dataset8 = tf.data.Dataset.from_tensor_slices(['hello', 'world'])
    input_dict8 = {"dataset_or_iterator": dataset8}
    list_of_inputs.append(input_dict8)

    # Input 9: Empty Dataset
    dataset9 = tf.data.Dataset.from_tensor_slices([])
    input_dict9 = {"dataset_or_iterator": dataset9}
    list_of_inputs.append(input_dict9)

    # Input 10: Dataset from a single tensor
    dataset10 = tf.data.Dataset.from_tensors(np.array([1, 2, 3]))
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
