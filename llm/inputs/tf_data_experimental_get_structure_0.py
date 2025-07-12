
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_get_structure_inputs():
    list_of_inputs = []

    # Input 1: Dataset from a single tensor
    dataset1 = tf.data.Dataset.from_tensor_slices(np.array([1, 2, 3]))
    input_dict1 = {"dataset_or_iterator": dataset1}
    list_of_inputs.append(input_dict1)

    # Input 2: Dataset from a tuple of tensors
    dataset2 = tf.data.Dataset.from_tensor_slices((np.array([1, 2, 3]), np.array(['a', 'b', 'c'])))
    input_dict2 = {"dataset_or_iterator": dataset2}
    list_of_inputs.append(input_dict2)

    # Input 3: Iterator from a Dataset (single tensor)
    dataset7 = tf.data.Dataset.from_tensor_slices(np.array([1, 2, 3]))
    iterator7 = iter(dataset7)
    input_dict7 = {"dataset_or_iterator": iterator7}
    list_of_inputs.append(input_dict7)

    # Input 4: Dataset from a tensor with negative values
    dataset9 = tf.data.Dataset.from_tensor_slices(np.array([-1, 0, 1]))
    input_dict9 = {"dataset_or_iterator": dataset9}
    list_of_inputs.append(input_dict9)

    # Input 5: Dataset from a 3D tensor
    dataset10 = tf.data.Dataset.from_tensor_slices(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    input_dict10 = {"dataset_or_iterator": dataset10}
    list_of_inputs.append(input_dict10)

    # Input 6: Dataset from a single string tensor
    dataset11 = tf.data.Dataset.from_tensor_slices(np.array(["hello", "world"]))
    input_dict11 = {"dataset_or_iterator": dataset11}
    list_of_inputs.append(input_dict11)

    # Input 7: Dataset from a tf.constant tensor
    tensor = tf.constant([1, 2, 3])
    dataset12 = tf.data.Dataset.from_tensor_slices(tensor)
    input_dict12 = {"dataset_or_iterator": dataset12}
    list_of_inputs.append(input_dict12)

    # Input 8: Dataset from a tuple with tf.constant
    tensor1 = tf.constant([1, 2, 3])
    tensor2 = tf.constant(['a', 'b', 'c'])
    dataset13 = tf.data.Dataset.from_tensor_slices((tensor1, tensor2))
    input_dict13 = {"dataset_or_iterator": dataset13}
    list_of_inputs.append(input_dict13)

    # Input 9: Iterator with constant tensors
    tensor1 = tf.constant([1, 2, 3])
    dataset14 = tf.data.Dataset.from_tensor_slices(tensor1)
    iterator14 = iter(dataset14)
    input_dict14 = {"dataset_or_iterator": iterator14}
    list_of_inputs.append(input_dict14)

    # Input 10: Dataset from tensor with rank 4
    dataset15 = tf.data.Dataset.from_tensor_slices(np.random.rand(2, 3, 4, 5).astype(np.float32))
    input_dict15 = {"dataset_or_iterator": dataset15}
    list_of_inputs.append(input_dict15)
    
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
