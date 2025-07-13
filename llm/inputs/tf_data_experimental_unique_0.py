
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_unique_inputs():
    list_of_inputs = []

    # Example 1: Simple list of integers
    dataset = np.array([1, 2, 3, 1, 2])
    input_dict = {"dataset": tf.data.Dataset.from_tensor_slices(dataset)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: List with duplicates
    dataset = np.array([4, 5, 6, 5, 4, 7])
    input_dict = {"dataset": tf.data.Dataset.from_tensor_slices(dataset)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: List of strings
    dataset = np.array(["a", "b", "c", "a", "b"])
    input_dict = {"dataset": tf.data.Dataset.from_tensor_slices(dataset)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4: List of floats
    dataset = np.array([1.0, 2.0, 3.0, 1.0, 2.0])
    input_dict = {"dataset": tf.data.Dataset.from_tensor_slices(dataset)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 5: List of tuples
    dataset = np.array([(1, 2), (3, 4), (1, 2)], dtype=object)
    input_dict = {"dataset": tf.data.Dataset.from_tensor_slices(dataset)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 6: List of arrays
    dataset = np.array([np.array([1, 2]), np.array([3, 4]), np.array([1, 2])], dtype=object)
    input_dict = {"dataset": tf.data.Dataset.from_tensor_slices(dataset)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 7: Empty list
    dataset = np.array([])
    input_dict = {"dataset": tf.data.Dataset.from_tensor_slices(dataset)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 8: Mixed types (but still hashable)
    dataset = np.array([1, "a", 2.0, 1, "a"], dtype=object)
    input_dict = {"dataset": tf.data.Dataset.from_tensor_slices(dataset)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 9: Large dataset
    dataset = np.random.randint(0, 10, size=100)
    input_dict = {"dataset": tf.data.Dataset.from_tensor_slices(dataset)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 10: Dataset with negative values
    dataset = np.array([-1, -2, -3, -1, -2])
    input_dict = {"dataset": tf.data.Dataset.from_tensor_slices(dataset)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.data.experimental.unique"] = tf_data_experimental_unique_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.data.experimental.unique' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.unique'.")

check_valid('tf.data.experimental.unique', generated_inputs['tf.data.experimental.unique'], lib="tf", suffix=0)
