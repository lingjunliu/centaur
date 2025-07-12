
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_data_experimental_to_variant_inputs():
    list_of_inputs = []

    # Input 1: Empty dataset
    dataset = tf.data.Dataset.from_tensor_slices(np.array([]))
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 2: Dataset with a single element
    dataset = tf.data.Dataset.from_tensor_slices(np.array([1]))
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 3: Dataset with multiple elements
    dataset = tf.data.Dataset.from_tensor_slices(np.array([1, 2, 3, 4, 5]))
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 4: Dataset with strings
    dataset = tf.data.Dataset.from_tensor_slices(np.array(["a", "b", "c"]))
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 5: Dataset with only integers in tuples, ensuring consistent types
    dataset = tf.data.Dataset.from_tensor_slices(np.array([(1, 2), (3, 4)]))
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 6: Dataset created from a range
    dataset = tf.data.Dataset.range(10)
    dataset = tf.data.Dataset.from_tensor_slices(np.array(list(dataset.as_numpy_iterator())))
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 7: Dataset created from a tensor
    tensor = tf.constant([[1, 2], [3, 4]])
    dataset = tf.data.Dataset.from_tensor_slices(tensor)
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 8: Dataset with a map function
    dataset = tf.data.Dataset.from_tensor_slices(np.array([1, 2, 3])).map(lambda x: tf.cast(x * 2, tf.int64)) #Explicitly cast to avoid errors in some backends.
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 9: Dataset with a filter function
    dataset = tf.data.Dataset.from_tensor_slices(np.array([1, 2, 3, 4, 5])).filter(lambda x: x % 2 == 0)
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 10: Dataset from numpy array with different shape

    dataset = tf.data.Dataset.from_tensor_slices(np.array([[1, 2, 3],[4,5,6]]))
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
