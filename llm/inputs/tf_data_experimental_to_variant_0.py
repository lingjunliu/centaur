
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_data_experimental_to_variant_inputs():
    list_of_inputs = []

    # Helper function to convert Dataset to numpy array
    def dataset_to_numpy(dataset):
        numpy_list = []
        for element in dataset:
            if isinstance(element, tf.Tensor):
                numpy_list.append(element.numpy())
            else:
                numpy_list.append(np.array(element))
        return np.array(numpy_list)

    # Input 1: Empty dataset
    dataset = tf.data.Dataset.from_tensor_slices([])
    input_dict = {"dataset": tf.constant(np.array([]))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Dataset with integers
    dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])
    numpy_array = dataset_to_numpy(dataset)
    input_dict = {"dataset": tf.constant(numpy_array)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Dataset with strings
    dataset = tf.data.Dataset.from_tensor_slices(["a", "b", "c"])
    numpy_array = dataset_to_numpy(dataset)
    input_dict = {"dataset": tf.constant(numpy_array)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Dataset with floats
    dataset = tf.data.Dataset.from_tensor_slices([1.0, 2.0, 3.0])
    numpy_array = dataset_to_numpy(dataset)
    input_dict = {"dataset": tf.constant(numpy_array)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    # Input 5: Dataset with tf.int64 data type
    dataset = tf.data.Dataset.from_tensor_slices(tf.constant([1, 2, 3], dtype=tf.int64))
    numpy_array = dataset_to_numpy(dataset)
    input_dict = {"dataset": tf.constant(numpy_array)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    # Input 6: Dataset with multi-dimensional tensors
    dataset = tf.data.Dataset.from_tensor_slices([tf.constant([[1, 2], [3, 4]]), tf.constant([[5, 6], [7, 8]])])
    numpy_array = dataset_to_numpy(dataset)
    input_dict = {"dataset": tf.constant(numpy_array)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Dataset with different length sequences, but convert to numpy array first
    list_data = [[1, 2], [3, 4, 5]]
    max_len = max(len(x) for x in list_data)
    padded_list = [x + [0] * (max_len - len(x)) for x in list_data]
    numpy_array = np.array(padded_list)
    dataset = tf.data.Dataset.from_tensor_slices(numpy_array)
    input_dict = {"dataset": tf.constant(numpy_array)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Dataset containing only float32 tensors
    dataset = tf.data.Dataset.from_tensor_slices([tf.constant([1.0, 2.0], dtype=tf.float32), tf.constant([3.0, 4.0], dtype=tf.float32)])
    numpy_array = dataset_to_numpy(dataset)
    input_dict = {"dataset": tf.constant(numpy_array)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Dataset of booleans
    dataset = tf.data.Dataset.from_tensor_slices([True, False, True])
    numpy_array = dataset_to_numpy(dataset)
    input_dict = {"dataset": tf.constant(numpy_array)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Dataset with 2D array of boolean
    dataset = tf.data.Dataset.from_tensor_slices([[True, False], [False, True]])
    numpy_array = dataset_to_numpy(dataset)
    input_dict = {"dataset": tf.constant(numpy_array)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Dataset containing tensors of different dtypes.
    dataset = tf.data.Dataset.from_tensor_slices([1, 2.0, "string"])
    
    def convert_to_string(element):
      return tf.strings.as_string(element)

    dataset = dataset.map(convert_to_string)
    numpy_array = dataset_to_numpy(dataset)
    input_dict = {"dataset": tf.constant(numpy_array)}
    list_of_inputs.append(copy.deepcopy(input_dict))

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
