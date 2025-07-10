
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_data_experimental_cardinality_inputs():
    list_of_inputs = []

    # Input 1: Dataset with a known finite cardinality (converted to numpy array)
    dataset1 = tf.data.Dataset.range(10)
    input_dict1 = {"dataset": np.array([i for i in dataset1.as_numpy_iterator()])}
    list_of_inputs.append(input_dict1)

    # Input 2: Dataset with a different known finite cardinality (converted to numpy array)
    dataset2 = tf.data.Dataset.range(100)
    input_dict2 = {"dataset": np.array([i for i in dataset2.as_numpy_iterator()])}
    list_of_inputs.append(input_dict2)

    # Input 3: Dataset created from a list (converted to numpy array)
    dataset3 = tf.data.Dataset.from_tensor_slices(tf.constant([1, 2, 3, 4, 5]))
    input_dict3 = {"dataset": np.array([i for i in dataset3.as_numpy_iterator()])}
    list_of_inputs.append(input_dict3)

    # Input 4: Dataset created from a tuple (converted to numpy array)
    dataset4 = tf.data.Dataset.from_tensor_slices(tf.constant([1, 2, 3]))
    input_dict4 = {"dataset": np.array([i for i in dataset4.as_numpy_iterator()])}
    list_of_inputs.append(input_dict4)

    # Input 5: Dataset created from a numpy array
    dataset5 = tf.data.Dataset.from_tensor_slices(np.array([1, 2, 3, 4]))
    input_dict5 = {"dataset": np.array([i for i in dataset5.as_numpy_iterator()])}
    list_of_inputs.append(input_dict5)

    # Input 7: Dataset that is filtered (cardinality might be unknown - converted to numpy array)
    dataset7 = tf.data.Dataset.range(10).filter(lambda x: x > 5)
    input_dict7 = {"dataset": np.array([i for i in dataset7.as_numpy_iterator()])}
    list_of_inputs.append(input_dict7)

    # Input 8: Dataset that is mapped (cardinality should be the same - converted to numpy array)
    dataset8 = tf.data.Dataset.range(5).map(lambda x: x * 2)
    input_dict8 = {"dataset": np.array([i for i in dataset8.as_numpy_iterator()])}
    list_of_inputs.append(input_dict8)

    # Input 9: Dataset that is batched (cardinality is number of batches - converted to numpy array)
    dataset9 = tf.data.Dataset.range(12).batch(4)
    input_dict9 = {"dataset": np.array([i for i in dataset9.as_numpy_iterator()])}
    list_of_inputs.append(input_dict9)

    # Input 10: Dataset from tensor slices with 2D array (converted to numpy array)
    dataset10 = tf.data.Dataset.from_tensor_slices(tf.constant([[1, 2], [3, 4], [5, 6]]))
    input_dict10 = {"dataset": np.array([i for i in dataset10.as_numpy_iterator()])}
    list_of_inputs.append(input_dict10)

    #Input 11: Empty numpy array
    input_dict11 = {"dataset": np.array([])}
    list_of_inputs.append(input_dict11)
    
    # Input 12: 1D numpy array
    input_dict12 = {"dataset": np.array([1, 2, 3])}
    list_of_inputs.append(input_dict12)

    # Input 13: 2D numpy array
    input_dict13 = {"dataset": np.array([[1, 2], [3, 4]])}
    list_of_inputs.append(input_dict13)


    return list_of_inputs

generated_inputs["tf.data.experimental.cardinality"] = tf_data_experimental_cardinality_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.data.experimental.cardinality' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.cardinality'.")

check_valid('tf.data.experimental.cardinality', generated_inputs['tf.data.experimental.cardinality'], lib="tf", suffix=0)
