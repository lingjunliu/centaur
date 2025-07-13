
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_data_experimental_cardinality_inputs():
    list_of_inputs = []

    # Helper function to convert dataset to a tensor (if possible)
    def dataset_to_tensor(dataset):
        try:
            return tf.constant(list(dataset.as_numpy_iterator()))
        except:
            return None

    # Input 1: Dataset with a known cardinality
    dataset = tf.data.Dataset.range(10)
    tensor = dataset_to_tensor(dataset)
    if tensor is not None:
        input_dict = {"dataset": tensor.numpy()}
        list_of_inputs.append(input_dict)

    # Input 2: Dataset with a different known cardinality
    dataset = tf.data.Dataset.range(100)
    tensor = dataset_to_tensor(dataset)
    if tensor is not None:
        input_dict = {"dataset": tensor.numpy()}
        list_of_inputs.append(input_dict)


    # Input 3: Dataset from tensor slices (known cardinality)
    tensor = tf.constant([1, 2, 3, 4, 5])
    input_dict = {"dataset": tensor.numpy()}
    list_of_inputs.append(input_dict)

    # Input 4: Dataset from tensors (known cardinality of 1)
    tensor = tf.constant([1, 2, 3])
    input_dict = {"dataset": tensor.numpy()}
    list_of_inputs.append(input_dict)

    # Input 5: Scalar tensor
    tensor = tf.constant(5)
    input_dict = {"dataset": tensor.numpy()}
    list_of_inputs.append(input_dict)

    # Input 6: Dataset from numpy array
    np_array = np.array([1, 2, 3, 4, 5])
    input_dict = {"dataset": np_array}
    list_of_inputs.append(input_dict)

    # Input 7: numpy array
    np_array = np.array([[1, 2], [3, 4]])
    input_dict = {"dataset": np_array}
    list_of_inputs.append(input_dict)

    # Input 8: numpy array with different dtype
    np_array = np.array([1.0, 2.0, 3.0])
    input_dict = {"dataset": np_array}
    list_of_inputs.append(input_dict)
    
    # Input 9: zero dimensional numpy array
    np_array = np.array(10)
    input_dict = {"dataset": np_array}
    list_of_inputs.append(input_dict)

    # Input 10: empty numpy array
    np_array = np.array([])
    input_dict = {"dataset": np_array}
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
