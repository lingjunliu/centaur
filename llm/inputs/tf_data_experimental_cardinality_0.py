
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_data_experimental_cardinality_inputs():
    list_of_inputs = []

    # Input 1: Dataset with a known cardinality
    dataset = tf.data.Dataset.from_tensor_slices(np.array([0,1,2,3,4,5,6,7,8,9]))
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 2: Dataset with a larger known cardinality
    dataset = tf.data.Dataset.from_tensor_slices(np.array([i for i in range(100)]))
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 3: Dataset created from a list
    dataset = tf.data.Dataset.from_tensor_slices(np.array([[1], [2], [3], [4], [5]]))
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 4: Dataset created from a tuple
    dataset = tf.data.Dataset.from_tensor_slices(np.array([[6], [7], [8], [9], [10]]))
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 5: Dataset created from a numpy array
    dataset = tf.data.Dataset.from_tensor_slices(np.array([[11], [12], [13], [14], [15]]))
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 6: Dataset with a map transformation, maintaining cardinality
    dataset = tf.data.Dataset.from_tensor_slices(np.array([0,1,2,3,4])).map(lambda x: x * 2)
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 7: Dataset with take, reducing cardinality
    dataset = tf.data.Dataset.from_tensor_slices(np.array([i for i in range(10)])).take(5)
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 8: Dataset with skip, maintaining cardinality if skip < cardinality, otherwise 0
    dataset = tf.data.Dataset.from_tensor_slices(np.array([i for i in range(10)])).skip(2)
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 9: Dataset with shuffle
    dataset = tf.data.Dataset.from_tensor_slices(np.array([i for i in range(10)])).shuffle(buffer_size=10)
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 10: Dataset with a simple filter
    dataset = tf.data.Dataset.from_tensor_slices(np.array([i for i in range(10)])).filter(lambda x: x < 5)
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.cardinality"] = tf_data_experimental_cardinality_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.data.experimental.cardinality' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.cardinality'.")

check_valid('tf.data.experimental.cardinality', generated_inputs['tf.data.experimental.cardinality'], lib="tf", suffix=0)
