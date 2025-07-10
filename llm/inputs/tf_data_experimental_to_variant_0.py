
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_data_experimental_to_variant_inputs():
    list_of_inputs = []

    # Input 1: Simple dataset
    dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 2: Dataset with a tuple of tensors
    dataset = tf.data.Dataset.from_tensor_slices(([1, 2, 3], [4, 5, 6]))
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 3: Dataset with a dictionary of tensors
    dataset = tf.data.Dataset.from_tensor_slices({"a": [1, 2, 3], "b": [4, 5, 6]})
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 4: Empty dataset
    dataset = tf.data.Dataset.from_tensor_slices([])
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 5: Dataset with different data types
    dataset = tf.data.Dataset.from_tensor_slices([1.0, 2.0, 3.0])
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 6: Dataset with strings
    dataset = tf.data.Dataset.from_tensor_slices(["a", "b", "c"])
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 7: Dataset with boolean values
    dataset = tf.data.Dataset.from_tensor_slices([True, False, True])
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 8: Dataset with multiple dimensions
    dataset = tf.data.Dataset.from_tensor_slices([[1, 2], [3, 4], [5, 6]])
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 9: Dataset created from a range
    dataset = tf.data.Dataset.range(10)
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)
    
    # Input 10: Scalar tensor
    dataset = tf.data.Dataset.from_tensors(tf.constant(1))
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.to_variant"] = tf_data_experimental_to_variant_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.data.experimental.to_variant' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.to_variant'.")

check_valid('tf.data.experimental.to_variant', generated_inputs['tf.data.experimental.to_variant'], lib="tf", suffix=0)
